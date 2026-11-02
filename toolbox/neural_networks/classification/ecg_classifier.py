import json
import os
from pathlib import Path
import socket


def ecg_classifier(ecg_path: str) -> list[str]:
    """Return findings above 0.5, sorted by probability, as 'LABEL (percentage)'."""
    endpoint = os.environ["EKO_CLASSIFIER"]
    request = {"ecg_path": str(Path(ecg_path).expanduser().resolve())}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(float(os.environ.get("EKO_PYTHON_TIMEOUT", "30")))
        connection.connect(endpoint)
        connection.sendall((json.dumps(request) + "\n").encode())
        with connection.makefile("rb") as reader:
            response = json.loads(reader.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]
