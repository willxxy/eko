"""Host Eko's model connection, sandbox, and terminal presentation."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable

import eko as core
import providers
from ui import UI


# ── Agent process and model service ──────────────────────────────────────────

class AgentProcess:
    """A local proxy for an eko.py process."""

    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.observer: Callable[[core.Event], None] = lambda event: None
        self.state = "starting"
        self.ready = threading.Event()
        self.session: str | None = None
        self.write_lock = threading.Lock()
        self.diagnostic_lock = threading.Lock()
        self.startup_diagnostics: list[str] = []
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.errors = threading.Thread(target=self.read_errors, daemon=True)
        self.reader.start()
        self.errors.start()

    def read(self) -> None:
        assert self.proc.stdout
        try:
            for line in self.proc.stdout:
                raw = json.loads(line)
                event = core.decode_event(raw)
                kind = event.type
                if kind == "ready":
                    self.session = raw["session"]
                    self.ready.set()
                else:
                    if kind == "state":
                        self.state = str(event.value)
                    self.observer(event)
        except Exception as error:
            message = f"Agent connection failed: {error}"
            self.record_startup_diagnostic(message)
            self.observer(core.Event("error", message))
        finally:
            self.ready.set()

    def read_errors(self) -> None:
        assert self.proc.stderr
        for line in self.proc.stderr:
            message = line.rstrip()
            self.record_startup_diagnostic(message)
            self.observer(core.Event("error", message))

    def record_startup_diagnostic(self, message: str) -> None:
        if self.session is None and message:
            with self.diagnostic_lock:
                self.startup_diagnostics.append(message)

    def startup_error(self) -> str:
        """Return diagnostics emitted before the agent became ready."""
        if self.proc.poll() is not None:
            self.reader.join(.2)
            self.errors.join(.2)
        with self.diagnostic_lock:
            details = "\n".join(self.startup_diagnostics).strip()
        return "agent did not start" + (f":\n{details}" if details else "")

    def send_command(self, value: dict) -> None:
        assert self.proc.stdin
        with self.write_lock:
            self.proc.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()

    def send(self, text: str) -> None:
        self.send_command({"type": "input", "content": [{"type": "text", "text": text}]})

    def interrupt(self) -> None:
        self.send_command({"type": "interrupt"})

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send_command({"type": "stop"})
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        self.reader.join(2)
        self.errors.join(2)

    def status(self) -> str:
        return self.state


def model_client(connection: socket.socket, backend: providers.Model) -> None:
    """Give one agent connection an independent model conversation."""
    lock = threading.Lock()
    active: threading.Thread | None = None

    def send(value: dict) -> None:
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        with lock:
            try:
                connection.sendall(data)
            except OSError:
                pass

    def complete(message: dict) -> None:
        try:
            reply = backend.complete(
                system, core.decode_message(message),
                lambda text: send({"delta": text}))
            send({"message": core.encode_message(reply),
                  "context_used": backend.context_used})
        except InterruptedError:
            send({"error": "Interrupted", "interrupted": True})
        except Exception as error:
            send({"error": str(error)})

    try:
        with connection, connection.makefile("rb") as reader:
            try:
                initial = json.loads(reader.readline())
                if not isinstance(initial, dict):
                    return
                system = initial.get("system")
                if not isinstance(system, str):
                    return
                while line := reader.readline():
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        return
                    if request.get("interrupt") is True:
                        backend.interrupt()
                    elif (isinstance(request.get("message"), dict) and
                          (active is None or not active.is_alive())):
                        active = threading.Thread(
                            target=complete, args=(request["message"],), daemon=True)
                        active.start()
                    else:
                        send({"error": "model generation already active"})
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
    finally:
        backend.interrupt()
        if active is not None:
            active.join(2)
        backend.close()


class ModelServer:
    def __init__(self, path: Path, factory: providers.Factory) -> None:
        self.path = path
        self.factory = factory
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        self.listener.listen()
        self.listener.settimeout(.2)
        self.stopping = threading.Event()
        self.clients: list[threading.Thread] = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        while not self.stopping.is_set():
            try:
                connection, address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            try:
                backend = self.factory()
            except Exception as error:
                connection.close()
                print(f"! Model initialization failed: {error}",
                      file=sys.stderr, flush=True)
                continue
            thread = threading.Thread(
                target=model_client,
                args=(connection, backend), daemon=True)
            thread.start()
            self.clients.append(thread)

    def close(self) -> None:
        self.stopping.set()
        self.listener.close()
        self.thread.join(2)
        for thread in self.clients:
            thread.join(2)


class WorldRelay:
    """Expose one upstream Unix stream socket at an agent-local path."""

    def __init__(self, path: Path, upstream: Path) -> None:
        self.path = path.resolve()
        self.upstream = upstream.expanduser().resolve()
        if self.path == self.upstream:
            raise ValueError("world relay endpoints must be different")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.path.chmod(0o600)
        self.listener.listen()
        self.listener.settimeout(.2)
        self.stopping = threading.Event()
        self.connections: set[socket.socket] = set()
        self.connection_lock = threading.Lock()
        self.clients: list[threading.Thread] = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def run(self) -> None:
        while not self.stopping.is_set():
            try:
                downstream, address = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            thread = threading.Thread(
                target=self.bridge, args=(downstream,), daemon=True
            )
            self.clients.append(thread)
            thread.start()

    def bridge(self, downstream: socket.socket) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sockets = (downstream, upstream)
        with self.connection_lock:
            self.connections.update(sockets)
        try:
            upstream.connect(str(self.upstream))
            threads = [
                threading.Thread(
                    target=self.pump, args=(source, destination), daemon=True
                )
                for source, destination in ((downstream, upstream), (upstream, downstream))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except OSError:
            pass
        finally:
            with self.connection_lock:
                self.connections.difference_update(sockets)
            for connection in sockets:
                connection.close()

    @staticmethod
    def pump(source: socket.socket, destination: socket.socket) -> None:
        try:
            while data := source.recv(64 * 1024):
                destination.sendall(data)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def close(self) -> None:
        self.stopping.set()
        self.listener.close()
        with self.connection_lock:
            connections = tuple(self.connections)
        for connection in connections:
            connection.close()
        self.thread.join(2)
        for thread in self.clients:
            thread.join(2)
        self.path.unlink(missing_ok=True)


def mount_parents(path: Path) -> list[str]:
    parents = list(path.parents)[:-1]
    return [item for parent in reversed(parents)
            for item in ("--dir", str(parent))]


def agent_command(cwd: Path, runtime: Path, *, sandbox: bool,
                  feral: bool, name: str, context: int = 0,
                  python_timeout: float = core.PYTHON_TIMEOUT,
                  clean_workspace: bool = False,
                  classifier: bool = False) -> list[str]:
    source = Path(core.__file__).resolve()
    arguments = ["--cwd", "/workspace" if sandbox else str(cwd),
                 "--model-socket",
                 "/run/eko/model.sock" if sandbox else str(runtime / "model.sock"),
                 "--session-socket",
                 "/run/eko/session.sock" if sandbox else str(runtime / "session.sock"),
                 "--name", name, "--python-timeout", str(python_timeout)]
    if context:
        arguments.extend(("--context", str(context)))
    if clean_workspace:
        arguments.append("--clean-workspace")
    if feral:
        arguments.append("--feral")
    if not sandbox:
        return [sys.executable, str(source), *arguments]
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("--sandbox requires Bubblewrap (bwrap)")
    environment = Path(sys.prefix).resolve()
    interpreter = Path(sys.executable).resolve()
    base = interpreter.parents[1]
    package_paths = []
    for name in ("purelib", "platlib"):
        path = Path(sysconfig.get_path(name)).resolve()
        try:
            relative = path.relative_to(environment)
        except ValueError as error:
            raise RuntimeError(
                f"sandbox package directory is outside the environment: {path}"
            ) from error
        mapped = str(Path("/opt/eko") / relative)
        if mapped not in package_paths:
            package_paths.append(mapped)
    if classifier:
        package_paths.append("/run/eko")
    return [
        bwrap, "--die-with-parent", "--new-session", "--as-pid-1",
        "--unshare-user", "--unshare-pid", "--unshare-ipc",
        "--unshare-uts", "--unshare-cgroup", "--unshare-net",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
        "--dir", "/etc", "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        *mount_parents(base), "--ro-bind", str(base), str(base),
        "--ro-bind", str(environment), "/opt/eko", "--dir", "/run",
        "--ro-bind", str(source), "/run/eko.py",
        "--bind", str(cwd), "/workspace", "--bind", str(runtime), "/run/eko",
        *(["--ro-bind", str(source.parent / "toolbox"), "/run/eko/toolbox"]
          if classifier else []),
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--remount-ro", "/", "--setenv", "HOME", "/workspace",
        "--setenv", "TMPDIR", "/tmp", "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PATH", "/opt/eko/bin:/usr/bin:/bin",
        "--setenv", "PYTHONPATH", os.pathsep.join(package_paths),
        "--setenv", "VIRTUAL_ENV", "/opt/eko",
        "--setenv", "EKO_MODEL", "/run/eko/model.sock",
        "--setenv", "EKO_WORLD", "/run/eko/world.sock",
        *(["--setenv", "EKO_CLASSIFIER", "/run/eko/classifier.sock"] if classifier else []),
        "--chdir", "/workspace", str(interpreter), "/run/eko.py",
        *arguments,
    ]


def print_event(event: core.Event) -> None:
    if event.type == "delta":
        print(event.value, end="", flush=True)
    elif event.type == "response":
        print(flush=True)
    elif event.type == "result":
        result = event.value
        if result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n",
                  flush=True)
    elif event.type == "error":
        print(f"! {event.value}", file=sys.stderr, flush=True)


def run(cwd: Path, prompt: str | None, *, model: str,
        feral: bool, name: str, headless: bool, sandbox: bool,
        world_socket: Path | None = None, context: int = 0,
        clean_workspace: bool = False,
        python_timeout: float = core.PYTHON_TIMEOUT,
        provider: str = "sglang", max_tokens: int = 4096,
        sglang_url: str | None = None,
        classifier_checkpoint: Path | None = None,
        classifier_device: str | None = None) -> None:
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"not a directory: {cwd}")
    factory = providers.factory(
        provider, model, max_tokens, sglang_url=sglang_url)
    classifier = None
    if classifier_checkpoint is not None:
        from toolbox.neural_networks.classification.model import Classifier

        classifier = Classifier(classifier_checkpoint, classifier_device)
    upstream_world = world_socket or os.environ.get("EKO_WORLD")
    with tempfile.TemporaryDirectory(prefix="eko-") as directory, ExitStack() as resources:
        runtime = Path(directory)
        if classifier is not None:
            from toolbox.neural_networks.classification.service import ClassifierServer

            classifier_server = ClassifierServer(
                runtime / "classifier.sock", classifier, cwd, sandbox=sandbox)
            classifier_server.start()
            resources.callback(classifier_server.close)
        server = ModelServer(runtime / "model.sock", factory)
        server.start()
        relay = (
            WorldRelay(runtime / "world.sock", Path(upstream_world))
            if upstream_world
            else None
        )
        if relay is not None:
            relay.start()
        environment = os.environ.copy()
        environment.pop("OPENROUTER_API_KEY", None)
        environment.pop("SGLANG_API_KEY", None)
        environment.pop("HF_TOKEN", None)
        environment.pop("EKO_CLASSIFIER", None)
        if classifier is not None:
            environment["EKO_CLASSIFIER"] = str(runtime / "classifier.sock")
            environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
                str(Path(__file__).resolve().parent), environment.get("PYTHONPATH"))))
        environment["EKO_WORLD"] = str(runtime / "world.sock")
        agent = AgentProcess(
            agent_command(
                cwd, runtime, sandbox=sandbox, feral=feral, name=name,
                context=context, clean_workspace=clean_workspace,
                python_timeout=python_timeout, classifier=classifier is not None
            ),
            # Bubblewrap's --setenv options supply the sandbox environment.
            env={} if sandbox else environment,
        )
        try:
            if not agent.ready.wait(10) or agent.proc.poll() is not None:
                raise RuntimeError(agent.startup_error())
            if headless:
                agent.observer = print_event
                print(f"EKO_SESSION={runtime / 'session.sock'}", flush=True)
                if prompt:
                    agent.send(prompt)
                while agent.proc.poll() is None:
                    time.sleep(.2)
                return
            ui = UI(context=context)
            ui.header(cwd, model, name)
            ui.connect(agent)
            ui.on_exit = lambda: (agent.stop(), ui.exit())
            ui.on_start = lambda: agent.send(prompt) if prompt else None
            if prompt:
                ui.user(prompt)
            ui.run()
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        finally:
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except KeyboardInterrupt:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            agent.stop()
            if relay is not None:
                relay.close()
            server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument(
        "--cwd", type=Path,
        help="workspace (feral mode defaults to a fresh temporary directory)",
    )
    parser.add_argument("--provider", default="sglang",
                        choices=("sglang", "openrouter", "huggingface"))
    parser.add_argument("--model", required=True,
                        help="model ID or local Hugging Face directory")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="output token limit (default: 4096)")
    parser.add_argument("--sglang-url",
                        help="SGLang server root URL (default: SGLANG_URL or "
                             "http://127.0.0.1:30002)")
    parser.add_argument("--name", default=core.NAME)
    parser.add_argument("--classifier-checkpoint", type=Path,
                        help="load a trusted fine-tuned ECG classifier once at startup")
    parser.add_argument("--classifier-device",
                        help="classifier device (default: CUDA when available, else CPU)")
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument(
        "--python-timeout", type=float, default=core.PYTHON_TIMEOUT,
        help=f"maximum seconds per Python action (default: {core.PYTHON_TIMEOUT:g})",
    )
    # Temporary experiment flag; remove after the workspace-hygiene evaluation.
    parser.add_argument("--clean-workspace", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument(
        "--feral", action="store_true",
        help="start immediately and keep acting autonomously",
    )
    parser.add_argument(
        "--world-socket", type=Path,
        help="connect the agent to an external world socket",
    )
    args = parser.parse_args()
    if args.python_timeout <= 0:
        parser.error("--python-timeout must be greater than zero")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.sglang_url is not None and args.provider != "sglang":
        parser.error("--sglang-url requires --provider sglang")
    def launch(cwd: Path) -> None:
        run(cwd, args.prompt, model=args.model,
            feral=args.feral, name=args.name, headless=args.headless,
            sandbox=args.sandbox, world_socket=args.world_socket,
            context=args.context,
            clean_workspace=args.clean_workspace,
            python_timeout=args.python_timeout,
            provider=args.provider, max_tokens=args.max_tokens,
            sglang_url=args.sglang_url,
            classifier_checkpoint=args.classifier_checkpoint,
            classifier_device=args.classifier_device)

    try:
        if args.feral and args.cwd is None:
            with tempfile.TemporaryDirectory(prefix="eko-workspace-") as directory:
                launch(Path(directory))
        else:
            launch(args.cwd or Path.cwd())
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
