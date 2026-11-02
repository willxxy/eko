"""ECG classification tool available to the agent."""

from .ecg_classifier import ecg_classifier

TOOLS = {"ecg_classifier": ecg_classifier}
