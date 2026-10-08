"""Private provider contract and in-memory conversation history."""

from __future__ import annotations

import threading
from typing import Callable, Protocol

import eko as core


class _Model(Protocol):
    context_used: int

    def complete(self, system: str, message: core.Message,
                 on_text: Callable[[str], None]) -> core.Message: ...

    def interrupt(self) -> None: ...

    def close(self) -> None: ...


_Factory = Callable[[], _Model]


class _Conversation:
    """Keep completed turns private to one agent connection."""

    def __init__(self) -> None:
        self.messages: list[core.Message] = []
        self.interrupted = threading.Event()
        self.context_used = 0

    def complete(self, system: str, message: core.Message,
                 on_text: Callable[[str], None]) -> core.Message:
        if message.role != "user":
            raise ValueError("model input must be a user message")
        self.interrupted.clear()
        messages = [*self.messages, message]
        text, used = self._generate(system, messages, on_text)
        if self.interrupted.is_set():
            raise InterruptedError
        if not text.strip():
            raise RuntimeError("Model produced no text")
        reply = core.Message("assistant", (core.Text(text),))
        self.messages = [*messages, reply]
        self.context_used = used
        return reply

    def _generate(self, system, messages, on_text) -> tuple[str, int]:
        raise NotImplementedError

    def interrupt(self) -> None:
        self.interrupted.set()

    def close(self) -> None:
        self.interrupt()
