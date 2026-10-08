"""Select a private model factory for the terminal host."""

from pathlib import Path

from .base import _Factory, _Model


def _factory(provider: str, model: str, max_tokens: int, *,
             cwd: Path, effort: str, session_id: str | None = None,
             resume: bool = False) -> _Factory:
    if provider == "claude":
        from .claude import _factory as create
        return create(cwd, model, effort, session_id, resume)
    if session_id or resume:
        raise ValueError("--session-id and --resume require --provider claude")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")
    if provider == "openrouter":
        from .openrouter import _factory as create
        return create(model, max_tokens)
    if provider == "huggingface":
        from .huggingface import _factory as create
        return create(model, max_tokens)
    raise ValueError(f"unknown provider: {provider}")
