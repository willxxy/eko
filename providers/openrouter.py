"""Streaming OpenRouter conversations."""

from __future__ import annotations

import base64
import http.client
import json
import os
import socket

import eko as core

from .base import _Conversation, _Factory


def _openrouter_message(message: core.Message) -> dict:
    content = []
    for part in message.content:
        if isinstance(part, core.Text):
            content.append({"type": "text", "text": part.text})
        else:
            if part.name:
                content.append({"type": "text", "text": f"Image: {part.name}"})
            data = base64.b64encode(part.data).decode()
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{part.media_type};base64,{data}"}})
    return {"role": message.role, "content": content}


class _OpenRouter(_Conversation):
    def __init__(self, model: str, api_key: str, max_tokens: int) -> None:
        super().__init__()
        self.model, self.api_key, self.max_tokens = model, api_key, max_tokens
        self.socket: socket.socket | None = None

    def _generate(self, system, messages, on_text) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         *map(_openrouter_message, messages)],
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        connection = http.client.HTTPSConnection("openrouter.ai", timeout=300)
        parts = []
        used = 0
        finish = None
        try:
            connection.connect()
            self.socket = connection.sock
            if self.interrupted.is_set():
                raise InterruptedError
            connection.request("POST", "/api/v1/chat/completions",
                               json.dumps(payload).encode(), {
                                   "Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json",
                               })
            with connection.getresponse() as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"OpenRouter HTTP {response.status}: "
                        f"{response.read().decode(errors='replace')}")
                for line in response:
                    if self.interrupted.is_set():
                        raise InterruptedError
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("error"):
                        raise RuntimeError(f"OpenRouter: {event['error']}")
                    if event.get("usage"):
                        used = int(event["usage"].get("prompt_tokens") or 0)
                    for choice in event.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            parts.append(text)
                            on_text(text)
                else:
                    raise RuntimeError("OpenRouter stream ended before [DONE]")
            if finish != "stop":
                raise RuntimeError(f"OpenRouter completion ended with {finish!r}")
            return "".join(parts), used
        except Exception:
            if self.interrupted.is_set():
                raise InterruptedError from None
            raise
        finally:
            self.socket = None
            connection.close()

    def interrupt(self) -> None:
        super().interrupt()
        connection = self.socket
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _factory(model: str, max_tokens: int) -> _Factory:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY for OpenRouter")
    return lambda: _OpenRouter(model, api_key, max_tokens)
