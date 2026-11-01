"""Serve the host's resident classifier over a private Unix socket."""

import json
from pathlib import Path
import socketserver
import threading


class ClassifierRequest(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(30)
        try:
            request = json.loads(self.rfile.readline(64 * 1024))
            path = Path(request["ecg_path"])
            if self.server.sandbox:
                path = self.server.cwd / path.relative_to("/workspace")
            elif not path.is_absolute():
                path = self.server.cwd / path
            path = path.resolve(strict=True)
            path.relative_to(self.server.cwd)
            response = {"result": self.server.classifier.evaluate(path)}
        except Exception as error:
            response = {"error": str(error)}
        try:
            self.wfile.write((json.dumps(response, allow_nan=False) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            pass


class ClassifierServer(socketserver.UnixStreamServer):
    def __init__(self, path: Path, classifier, cwd: Path, *, sandbox: bool) -> None:
        self.classifier = classifier
        self.cwd = cwd.resolve()
        self.sandbox = sandbox
        super().__init__(str(path), ClassifierRequest)
        path.chmod(0o600)
        self.thread = threading.Thread(target=self.serve_forever,
                                       kwargs={"poll_interval": .2}, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.shutdown()
        self.server_close()
        self.thread.join()
