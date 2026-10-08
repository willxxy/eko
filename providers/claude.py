"""Claude CLI conversations, authentication, and session recovery."""

from __future__ import annotations

import base64
import json
import os
import select
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import eko as core

from .base import _Factory


CALL_TIMEOUT = 300


def _claude_content(message: core.Message) -> list[dict]:
    """Serialize provider-neutral content as Claude blocks."""
    blocks: list[dict] = []
    for part in message.content:
        if isinstance(part, core.Text):
            blocks.append({"type": "text", "text": part.text})
        else:
            if part.name:
                blocks.append({"type": "text", "text": f"Image: {part.name}"})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": part.media_type,
                "data": base64.b64encode(part.data).decode(),
            }})
    return blocks


class Claude:
    """A persistent, tool-free connection to the LLM through the Claude CLI.

    Stream JSON lets several Eko turns share one model conversation. ``--safe-mode``
    prevents machine-specific instructions, hooks, plugins, and skills from changing
    the model's context, while ``--tools ''`` leaves generated Python as its only action.
    """

    def __init__(self, cwd: Path, model: str = "claude-opus-5",
                 effort: str = "high", session_id: str | None = None,
                 resume: bool = False) -> None:
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self.session_id = session_id or str(uuid.uuid4())
        uuid.UUID(self.session_id)
        self.proc: subprocess.Popen[bytes] | None = None
        self.started = resume
        self.interrupted = threading.Event()
        self.context_used = 0

    def _repair_session(self) -> bool:
        """Repair this session's empty assistant text blocks."""
        config = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
        projects = config / "projects"
        if not projects.is_dir():
            return False
        for path in projects.glob(f"*/{self.session_id}.jsonl"):
            lines = path.read_text().splitlines(keepends=True)
            changed = False
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") != "assistant":
                    continue
                content = record.get("message", {}).get("content", [])
                repaired = False
                for block in content:
                    if block.get("type") == "text" and block.get("text") == "":
                        block["text"] = " "
                        repaired = changed = True
                if repaired:
                    ending = "\n" if line.endswith("\n") else ""
                    lines[index] = json.dumps(record, separators=(",", ":")) + ending
            if changed:
                temporary = path.with_suffix(".jsonl.tmp")
                temporary.write_text("".join(lines))
                os.replace(temporary, path)
                return True
        return False

    def _start(self, system: str) -> None:
        session = (["--resume", self.session_id] if self.started else
                   ["--session-id", self.session_id])
        command = [
            "claude", "-p", "--verbose", "--safe-mode", "--tools", "",
            "--model", self.model, "--effort", self.effort,
            *session,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--include-partial-messages",
            "--system-prompt", system,
        ]
        self.proc = subprocess.Popen(
            command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        self.started = True

    def _terminate(self, signum: int, grace: float = 2) -> None:
        """Signal the CLI process group and ensure it is collected."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signum)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
        self.proc = None

    def complete(self, system: str, message: core.Message,
                 on_text: Callable[[str], None],
                 deadline: float | None = None,
                 retry_delay: float = .2) -> core.Message:
        """Complete a history using the CLI's internally persisted conversation."""
        if message.role != "user":
            raise ValueError("model input must be a user message")
        self.interrupted.clear()
        deadline = deadline or time.monotonic() + CALL_TIMEOUT
        resuming = self.started
        if self.proc is None or self.proc.poll() is not None:
            self._start(system)
        proc = self.proc
        assert proc is not None and proc.stdin and proc.stdout
        event = {"type": "user", "message": {
            "role": "user", "content": _claude_content(message)}}
        proc.stdin.write((json.dumps(event) + "\n").encode())
        proc.stdin.flush()

        parts: list[str] = []
        complete = ""
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [proc.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                if self.interrupted.is_set():
                    raise InterruptedError
                break
            if not line.startswith(b"{"):
                continue
            data = json.loads(line)
            if data.get("type") == "stream_event":
                event = data.get("event", {})
                delta = event.get("delta", {})
                if (event.get("type") == "content_block_delta"
                        and delta.get("type") == "text_delta"):
                    text = delta.get("text", "")
                    parts.append(text)
                    on_text(text)
            elif data.get("type") == "assistant":
                complete = "".join(
                    block["text"] for block in data["message"].get("content", [])
                    if block.get("type") == "text")
            elif data.get("type") == "result":
                if data.get("is_error"):
                    detail = data.get("result") or data.get("error")
                    if resuming:
                        self._terminate(signal.SIGTERM)
                        proc.stdin.close()
                        proc.stdout.close()
                        if ("text content blocks must be non-empty" in str(detail)
                                and self._repair_session()):
                            return self.complete(
                                system, message, on_text, deadline, retry_delay)
                        remaining = deadline - time.monotonic()
                        if remaining > 0 and not parts and not complete:
                            delay = min(retry_delay, remaining)
                            if self.interrupted.wait(delay):
                                raise InterruptedError
                            if time.monotonic() < deadline:
                                return self.complete(
                                    system, message, on_text, deadline,
                                    min(retry_delay * 2, 5))
                        raise RuntimeError(
                            "Model session could not resume; context was not "
                            f"reset. {detail or ''}".rstrip())
                    raise RuntimeError(detail or "Model call failed")
                usage = data.get("usage") or {}
                self.context_used = (int(usage["prompt_tokens"])
                                     if usage.get("prompt_tokens") is not None else
                                     sum(int(usage.get(name) or 0) for name in (
                                         "input_tokens", "cache_read_input_tokens",
                                         "cache_creation_input_tokens")))
                return core.Message(
                    "assistant", (core.Text(complete or "".join(parts)),))
        raise RuntimeError("Model produced no result")

    def close(self) -> None:
        """Give the CLI a brief chance to flush its session, then stop it."""
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                assert proc.stdin
                proc.stdin.close()
                proc.stdin = None
                proc.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._terminate(signal.SIGTERM)
                return
        if proc.stdout:
            proc.stdout.close()
        if self.proc is proc:
            self.proc = None

    def interrupt(self) -> None:
        self.interrupted.set()
        self._terminate(signal.SIGKILL)


# ── Claude authentication ─────────────────────────────────────────────────────

def auth_status() -> bool:
    """Return whether the Claude CLI can access the model."""
    if shutil.which("claude") is None:
        raise SystemExit("Claude Code is not installed or is not on PATH.")
    status = subprocess.run(
        ["claude", "auth", "status", "--json"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if status.returncode:
        return False
    try:
        data = json.loads(status.stdout)
        return bool(data.get("loggedIn"))
    except json.JSONDecodeError:
        return False


def ensure_auth() -> None:
    """Let the official CLI own sign-in; Eko never reads or stores credentials."""
    if auth_status():
        return
    print("Claude Code is not signed in.")
    answer = input("Press Enter to sign in, or q to exit: ").strip().lower()
    if answer == "q":
        raise SystemExit(0)
    subprocess.run(["claude", "auth", "login", "--claudeai"], check=False)
    if not auth_status():
        raise SystemExit("Claude sign-in did not complete.")


def _factory(cwd: Path, model: str, effort: str,
             session_id: str | None, resume: bool) -> _Factory:
    if session_id:
        uuid.UUID(session_id)
    ensure_auth()
    primary_session = (session_id, resume)
    lock = threading.Lock()

    def create() -> Claude:
        nonlocal primary_session
        with lock:
            assigned, resuming = primary_session
            primary_session = (None, False)
        return Claude(cwd, model, effort, assigned, resuming)

    return create
