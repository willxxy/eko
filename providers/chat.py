"""Shared streaming chat-completions conversations."""

from __future__ import annotations

import base64
import http.client
import json
import socket
from urllib.parse import urlsplit

import eko as core

from .base import Conversation


def chat_message(message: core.Message) -> dict:
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


class ChatCompletions(Conversation):
    def __init__(self, name: str, endpoint: str, model: str,
                 api_key: str | None, max_tokens: int) -> None:
        super().__init__()
        self.name = name
        self.url = urlsplit(endpoint)
        if (self.url.scheme not in ("http", "https") or not self.url.hostname or
                self.url.username or self.url.password or
                self.url.query or self.url.fragment):
            raise ValueError(f"{name} endpoint must be an HTTP(S) URL "
                             "without credentials, query, or fragment")
        self.model, self.api_key, self.max_tokens = model, api_key, max_tokens
        self.socket: socket.socket | None = None

    def generate(self, system, messages, on_text) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         *map(chat_message, messages)],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        connect = (http.client.HTTPSConnection if self.url.scheme == "https"
                   else http.client.HTTPConnection)
        connection = connect(self.url.hostname, self.url.port, timeout=300)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        parts = []
        used = 0
        finish = None
        try:
            connection.connect()
            self.socket = connection.sock
            if self.interrupted.is_set():
                raise InterruptedError
            connection.request("POST", self.url.path,
                               json.dumps(payload).encode(), headers)
            with connection.getresponse() as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"{self.name} HTTP {response.status}: "
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
                        raise RuntimeError(f"{self.name}: {event['error']}")
                    if event.get("usage"):
                        used = int(event["usage"].get("prompt_tokens") or 0)
                    for choice in event.get("choices", []):
                        finish = choice.get("finish_reason") or finish
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            parts.append(text)
                            on_text(text)
                else:
                    raise RuntimeError(f"{self.name} stream ended before [DONE]")
            if finish != "stop":
                raise RuntimeError(f"{self.name} completion ended with {finish!r}")
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
