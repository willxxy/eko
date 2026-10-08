"""Local Transformers conversations sharing one model instance."""

from __future__ import annotations

import threading

import eko as core

from .base import _Conversation, _Factory


class _LocalModel:
    """Load weights once; serialize generation across agent conversations."""

    def __init__(self, model: str, max_tokens: int) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Hugging Face requires transformers, torch, and accelerate; "
                "see README.md") from error
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.chat_template:
            raise ValueError("Hugging Face model must have a chat template")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype="auto")
        self.model.eval()
        self.max_tokens = max_tokens
        self.lock = threading.Lock()


class _HuggingFace(_Conversation):
    def __init__(self, runtime: _LocalModel) -> None:
        super().__init__()
        self.runtime = runtime

    def _generate(self, system, messages, on_text) -> tuple[str, int]:
        import torch
        from transformers import StoppingCriteriaList, TextStreamer

        chat = [{"role": "system", "content": system}]
        for message in messages:
            if any(isinstance(part, core.Image) for part in message.content):
                raise ValueError("Local Hugging Face models accept text only")
            chat.append({"role": message.role, "content": core.message_text(message)})

        class Stream(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False):
                if text:
                    on_text(text)

        runtime = self.runtime
        while not runtime.lock.acquire(timeout=.1):
            if self.interrupted.is_set():
                raise InterruptedError
        try:
            if self.interrupted.is_set():
                raise InterruptedError
            inputs = runtime.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to(runtime.model.device)
            used = inputs["input_ids"].shape[-1]
            streamer = Stream(runtime.tokenizer, skip_prompt=True,
                              skip_special_tokens=True)
            stopping = StoppingCriteriaList([
                lambda _ids, _scores, **_kwargs: self.interrupted.is_set()])
            with torch.inference_mode():
                output = runtime.model.generate(
                    **inputs, max_new_tokens=runtime.max_tokens,
                    do_sample=False, num_beams=1, num_return_sequences=1,
                    return_dict_in_generate=False, streamer=streamer,
                    stopping_criteria=stopping)
            if self.interrupted.is_set():
                raise InterruptedError
            tokens = output[0, used:]
            eos = runtime.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            if len(tokens) >= runtime.max_tokens and int(tokens[-1]) not in eos:
                raise RuntimeError("Hugging Face completion reached --max-tokens")
            return runtime.tokenizer.decode(tokens, skip_special_tokens=True), used
        finally:
            runtime.lock.release()


def _factory(model: str, max_tokens: int) -> _Factory:
    runtime = _LocalModel(model, max_tokens)
    return lambda: _HuggingFace(runtime)
