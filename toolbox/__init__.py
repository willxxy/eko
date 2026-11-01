"""Tools available to the agent."""

import json
import os
from pathlib import Path
import socket


def ecg_classifier(ecg_path: str) -> list[str]:
    """Classify a training-preprocessed 12-lead .mat in the agent's workspace.

    Returns ECG-Agent evaluation findings above 0.5, sorted by probability and
    formatted as "LABEL (percentage)". Requires host.py --classifier-checkpoint.
    """
    endpoint = os.environ.get("EKO_CLASSIFIER")
    if not endpoint:
        raise RuntimeError("Start host.py with --classifier-checkpoint")
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


TOOLS = {"ecg_classifier": ecg_classifier}
