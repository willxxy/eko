from __future__ import annotations
import threading
import eko as core
from .base import Conversation, Factory

THINK_START = "<think>"
THINK_END = "</think>"


class AnswerFilter:
    """Skip a leading thinking block, including tags split across chunks."""

    def __init__(self, *, prefilled: bool) -> None:
        self.thinking = prefilled
        self.pending: str | None = ""

    def feed(self, text: str, *, final: bool = False) -> str:
        if self.pending is None:
            return text

        text = self.pending + text
        if not self.thinking:
            prefix = text.lstrip()
            if prefix.startswith(THINK_START):
                self.thinking = True
            elif not final and THINK_START.startswith(prefix):
                self.pending = text
                return ""

        if self.thinking:
            _, end, answer = text.partition(THINK_END)
            if end:
                text = answer
            elif final:
                raise RuntimeError("Hugging Face completion ended inside a thinking block")
            else:
                self.pending = text[-(len(THINK_END) - 1):]
                return ""

        self.pending = None
        return text


class LocalModel:
    def __init__(self, model: str, max_tokens: int) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if not self.tokenizer.chat_template:
            raise ValueError("Hugging Face model must have a chat template")
        self.model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype="auto")
        self.model.eval()
        self.max_tokens = max_tokens
        self.lock = threading.Lock()

class HuggingFace(Conversation):
    def __init__(self, runtime: LocalModel) -> None:
        super().__init__()
        self.runtime = runtime

    def generate(self, system, messages, on_text) -> tuple[str, int]:
        import torch
        from transformers import StoppingCriteriaList, TextStreamer

        chat = [{"role": "system", "content": system}]
        for message in messages:
            if any(isinstance(part, core.Image) for part in message.content):
                raise ValueError("Local Hugging Face models accept text only")
            chat.append({"role": message.role, "content": core.message_text(message)})

        class Stream(TextStreamer):
            def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                if answer := stream_filter.feed(text):
                    on_text(answer)

        runtime = self.runtime
        while not runtime.lock.acquire(timeout=.1):
            if self.interrupted.is_set():
                raise InterruptedError
        try:
            if self.interrupted.is_set():
                raise InterruptedError
            tokenizer = runtime.tokenizer
            prompt = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True)
            prefilled = prompt.rstrip().endswith(THINK_START)
            inputs = tokenizer(prompt, add_special_tokens=False,
                               return_tensors="pt").to(runtime.model.device)
            used = inputs["input_ids"].shape[-1]
            stream_filter = AnswerFilter(prefilled=prefilled)
            streamer = Stream(tokenizer, skip_prompt=True, skip_special_tokens=True)
            stopping = StoppingCriteriaList([
                lambda ids, scores, **kwargs: self.interrupted.is_set()])
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
            # Full decoding can preserve whitespace that the streamer drops.
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            answer = AnswerFilter(prefilled=prefilled).feed(text, final=True)
            if tail := stream_filter.feed("", final=True):
                on_text(tail)
            return answer, used
        finally:
            runtime.lock.release()


def factory(model: str, max_tokens: int) -> Factory:
    runtime = LocalModel(model, max_tokens)
    return lambda: HuggingFace(runtime)
