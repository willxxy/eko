"""Streaming terminal output and prompt controls."""

from __future__ import annotations

import io
import shutil
import sys
import threading
import time
import tokenize
from pathlib import Path
from typing import Callable

import eko as core
from markdown_it import MarkdownIt
from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text as RichText


MAX_DISPLAY_OUTPUT = 4_000
MAX_DISPLAY_LINES = 5
GOLD = "#d7af5f"
GOLD_ACTIVE = "#e5bd68"
MARKDOWN_WITH_LITERAL_TAGS = MarkdownIt("commonmark", {"html": False}).enable(
    ["strikethrough", "table"])


def display_output(text: str, width: int = 120) -> str:
    """Make a compact terminal preview while preserving model-facing output."""
    lines = text.splitlines()
    if len(lines) > MAX_DISPLAY_LINES:
        retained = MAX_DISPLAY_LINES - 1
        head = retained // 2
        tail = retained - head
        omitted = len(lines) - retained
        lines = lines[:head] + [f"… +{omitted} lines"] + lines[-tail:]
    lines = [line if len(line) <= width else line[:width - 1] + "…"
             for line in lines]
    text = "\n".join(lines)
    if len(text) <= MAX_DISPLAY_OUTPUT:
        return text
    half = MAX_DISPLAY_OUTPUT // 2
    omitted = len(text) - MAX_DISPLAY_OUTPUT
    return (f"{text[:half]}\n\n… {omitted:,} characters omitted …\n\n"
            f"{text[-half:]}")


class NativeStream:
    """Append stable model-output lines without redrawing terminal history."""

    def __init__(self, ui: UI) -> None:
        self.ui = ui
        self.buffer = ""
        self.pending_output: list[str] = []
        self.last_flush = time.monotonic()
        self.fence_length = 0
        self.prose = ""
        self.code_lines: list[str] = []
        self.code_emitted = 0

    def feed(self, delta: str) -> None:
        self.buffer += delta.replace("\x1b", "")
        self.drain()
        if time.monotonic() - self.last_flush >= .1:
            self.flush()

    def flush(self) -> None:
        if self.pending_output:
            self.ui.append("".join(self.pending_output))
            self.pending_output.clear()
        self.last_flush = time.monotonic()

    def drain(self, final: bool = False) -> None:
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()
        for line in lines:
            self.line(line + "\n")
        if not final:
            return
        if self.buffer:
            line, self.buffer = self.buffer, ""
            self.line(line)
        if self.fence_length:
            self.close_box()
        else:
            self.prose = self.prose.rstrip().removesuffix(core.DONE_MARKER)
            self.add_prose("", final=True)

    def line(self, line: str) -> None:
        if self.fence_length:
            if core.closing_fence(line, self.fence_length):
                self.close_box()
            else:
                self.code_lines.append(line.rstrip("\r\n"))
                self.commit_code()
            return
        if length := core.opening_fence(line):
            self.add_prose("", final=True)
            self.fence_length = length
            self.open_box()
        else:
            self.add_prose(line)

    def add_prose(self, text: str, final: bool = False) -> None:
        self.prose += text
        # A blank line closes a Markdown block. Keeping only the unfinished
        # block mutable prevents later tokens from restyling terminal history.
        while "\n\n" in self.prose:
            block, rest = self.prose.split("\n\n", 1)
            if not final and block.rstrip().endswith(core.DONE_MARKER) and not rest.strip():
                break
            self.render_prose(block)
            self.prose = rest
        if final and self.prose:
            self.render_prose(self.prose)
            self.prose = ""

    def render_prose(self, text: str) -> None:
        text = text.strip()
        if text:
            markdown = Markdown(text)
            if core.DONE_MARKER in text:
                markdown.parsed = MARKDOWN_WITH_LITERAL_TAGS.parse(text)
            self.pending_output.append(self.ui.render(markdown) + "\n")

    def width(self) -> int:
        return max(20, shutil.get_terminal_size().columns - 2)

    def open_box(self) -> None:
        self.code_lines = []
        self.code_emitted = 0
        width = self.width()
        title = "─ python "
        self.pending_output.append(self.ui.render(RichText(
            "╭" + title + "─" * max(0, width - len(title) - 2) + "╮",
            style=f"bold {GOLD}")))

    def stable_code_lines(self) -> int:
        """Return the prefix whose Python highlighting cannot change later."""
        source = "\n".join(self.code_lines) + "\n"
        try:
            list(tokenize.generate_tokens(io.StringIO(source).readline))
        except tokenize.TokenError as error:
            line = error.args[1][0]
        except SyntaxError as error:
            line = error.lineno or 1
        else:
            return len(self.code_lines)
        return min(len(self.code_lines), max(0, line - 1))

    def commit_code(self, final: bool = False) -> None:
        end = len(self.code_lines) if final else self.stable_code_lines()
        if end <= self.code_emitted:
            return
        source = "\n".join(self.code_lines) + "\n"
        highlighted = Syntax("", "python", theme="ansi_dark").highlight(source)
        lines = highlighted.split("\n")
        for line in lines[self.code_emitted:end]:
            self.code_line(line)
        self.code_emitted = end

    def code_line(self, line: RichText) -> None:
        width = self.width()
        content = line[:max(1, width - 4)]
        padding = " " * max(0, width - len(content.plain) - 4)
        row = RichText()
        row.append("│ ", style=GOLD)
        row.append_text(content)
        row.append(padding)
        row.append(" │", style=GOLD)
        self.pending_output.append(self.ui.render(row))

    def close_box(self) -> None:
        self.commit_code(final=True)
        width = self.width()
        self.pending_output.append(self.ui.render(RichText(
            "╰" + "─" * (width - 2) + "╯", style=GOLD)))
        self.fence_length = 0

    def finish(self) -> None:
        self.drain(final=True)
        self.flush()


class UI:
    """Commit completed messages above a small live composer."""

    def __init__(self, context: int = 0) -> None:
        self.live = ""
        self.lock = threading.Lock()
        self.status: Callable[[], str] = lambda: "idle"
        self.context_capacity = context
        self.context_used = 0
        self.on_submit: Callable[[str], None] = lambda text: None
        self.on_interrupt: Callable[[], None] = lambda: None
        self.on_exit: Callable[[], None] = lambda: None
        self.on_start: Callable[[], None] = lambda: None
        self.keys = KeyBindings()

        self.transcript_window = Window(
            FormattedTextControl(self.transcript), height=1,
            wrap_lines=False, always_hide_cursor=True)
        self.composer = TextArea(
            height=1,
            prompt=[(f"{GOLD} bold", "› ")],
            multiline=True, wrap_lines=True)
        self.status_window = Window(FormattedTextControl(self.status_text), height=1)

        @self.keys.add("enter")
        def send(event) -> None:
            self.submit()

        @self.keys.add("c-c")
        def cancel(event) -> None:
            if self.composer.text:
                self.composer.buffer.set_document(Document("", 0), bypass_readonly=True)
            else:
                self.on_interrupt()

        @self.keys.add("c-d")
        def exit_app(event) -> None:
            if not self.composer.text:
                self.on_exit()

        @self.keys.add("escape")
        def interrupt(event) -> None:
            self.on_interrupt()

        root = HSplit([
            self.transcript_window,
            Window(height=1, char="─", style="class:rule"),
            self.composer,
            self.status_window,
        ])
        self.app: Application[None] = Application(
            layout=Layout(root, focused_element=self.composer),
            key_bindings=self.keys, full_screen=False, mouse_support=False,
            min_redraw_interval=.04,
            style=Style.from_dict({
                "rule": "#444444", "status": "#888888",
            }))
        self.app.ttimeoutlen = .1
        self.app.timeoutlen = .1

    def connect(self, agent) -> None:
        """Observe an Eko and route terminal controls back to it."""
        stream = None

        def render(event: core.Event) -> None:
            nonlocal stream
            if event.type == "state":
                state = str(event.value)
                if state == "thinking":
                    self.start_activity("thinking")
                elif state == "running Python":
                    self.start_activity("running")
                else:
                    self.stop_activity()
            elif event.type == "delta":
                if stream is None:
                    stream = NativeStream(self)
                stream.feed(str(event.value))
            elif event.type == "response":
                self.stop_activity()
                if stream is None:
                    stream = NativeStream(self)
                    stream.feed(event.value[0])
                stream.finish()
                stream = None
            elif event.type == "result":
                self.stop_activity()
                self.result(event.value)
            elif event.type == "context":
                self.context_used, self.context_capacity = map(int, event.value)
                self.app.invalidate()
            elif event.type == "error":
                self.stop_activity()
                if stream is not None:
                    stream.finish()
                    stream = None
                self.stopped(str(event.value))

        agent.observer = render
        self.status = agent.status
        self.on_submit = agent.send
        self.on_interrupt = agent.interrupt

    def start_activity(self, label: str) -> None:
        """Show current activity above the prompt."""
        with self.lock:
            self.live = self.render(RichText(label, style=f"dim {GOLD_ACTIVE}"))
        self.app.invalidate()

    def stop_activity(self) -> None:
        with self.lock:
            self.live = ""
        self.app.invalidate()

    def submit(self) -> None:
        text = self.composer.text.strip()
        if not text:
            return
        self.composer.buffer.set_document(Document("", 0), bypass_readonly=True)
        if text == "/exit":
            self.on_exit()
            return
        self.user(text)
        self.on_submit(text)

    def transcript(self):
        with self.lock:
            return ANSI(self.live)

    def status_text(self):
        state = self.status()
        active = state.startswith(("thinking", "running"))
        hint = ("esc interrupt · enter send" if active else
                "enter send · ctrl+d exit")
        context = (f" · {core.context_status_line(self.context_used, self.context_capacity)}"
                   if self.context_capacity else "")
        return [("class:status", f"  {hint}{context}")]

    def render(self, renderable) -> str:
        stream = io.StringIO()
        width = max(40, shutil.get_terminal_size().columns - 2)
        Console(file=stream, force_terminal=True, color_system="truecolor",
                highlight=False, width=width).print(renderable)
        return stream.getvalue()

    def append(self, text: str) -> None:
        def write() -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        if not self.app.is_running or self.app.loop is None:
            write()
        elif threading.current_thread() is self.app._loop_thread:
            run_in_terminal(write)
        else:
            done = threading.Event()

            def schedule() -> None:
                task = run_in_terminal(write)
                task.add_done_callback(lambda future: done.set())

            self.app.loop.call_soon_threadsafe(schedule)
            done.wait(2)

    def header(self, cwd: Path, model: str, name: str = core.NAME) -> None:
        header = Group(RichText(name, style=f"bold {GOLD}"), RichText.assemble(
            (str(cwd), "dim"), ("  ·  ", "dim"), (model, "dim"),
            ("  ·  signed in", "dim")), RichText(""))
        # Place the initial composer near the bottom. These are real terminal
        # lines, so subsequent turns naturally replace them and enter native
        # scrollback instead of making a full-screen spacer jump around.
        padding = max(
            0, shutil.get_terminal_size().lines - 9)
        self.append(self.render(header) + "\n" * padding)

    def user(self, text: str) -> None:
        rendered = self.render(RichText.assemble(("› ", f"bold {GOLD}"), text))
        self.append(rendered + "\n")

    def result(self, result: core.Result) -> None:
        ok = result.returncode == 0
        style = "dim" if ok else "red"
        label = f"Exit {result.returncode} · {result.elapsed:.1f}s"
        width = max(40, shutil.get_terminal_size().columns - 4)
        output = display_output(result.output, width)
        body = RichText(output.rstrip() or "(no output)", style="dim")
        self.append(self.render(Group(RichText(label, style=style), body)) + "\n")

    def stopped(self, message: str = "Stopped") -> None:
        rendered = self.render(RichText(f"! {message}", style="#ff875f")) + "\n"
        self.append(rendered)

    def run(self) -> None:
        self.app.pre_run_callables.append(self.on_start)
        self.app.run()

    def exit(self) -> None:
        if self.app.is_running:
            self.app.exit()
