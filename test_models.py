"""Provider contracts without credentials, model downloads, or a GPU."""

import io
import json
import socket
import subprocess
import sys
import threading
import unittest
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import eko
import host
import providers
from providers import claude, huggingface, openrouter


def message(text):
    return eko.Message("user", (eko.Text(text),))


def stream(*events, done=True, status=200):
    data = b": keepalive\n\n"
    for event in events:
        data += b"data: " + json.dumps(event).encode() + b"\n\n"
    if done:
        data += b"data: [DONE]\n\n"
    response = io.BytesIO(data)
    response.status = status
    return response


def chunk(text="", finish=None):
    return {"choices": [{"delta": {"content": text}, "finish_reason": finish}]}


class ClaudeTests(unittest.TestCase):
    def test_cli_keeps_isolation_streaming_and_session_options(self):
        session_id = "12345678-1234-5678-1234-567812345678"
        model = claude.Claude(Path.cwd(), session_id=session_id)
        with mock.patch.object(claude.subprocess, "Popen") as start:
            model._start("system")
            self.assertEqual(start.call_args.args[0], [
                "claude", "-p", "--verbose", "--safe-mode", "--tools", "",
                "--model", "claude-opus-5", "--effort", "high",
                "--session-id", session_id,
                "--input-format", "stream-json", "--output-format", "stream-json",
                "--include-partial-messages", "--system-prompt", "system",
            ])
            self.assertEqual(start.call_args.kwargs, {
                "cwd": Path.cwd(), "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                "bufsize": 0, "start_new_session": True,
            })
            model._start("system")
            command = start.call_args.args[0]
            self.assertNotIn("--session-id", command)
            self.assertEqual(command[command.index("--resume") + 1], session_id)

    def test_factory_reserves_persisted_session_for_first_conversation(self):
        session_id = "12345678-1234-5678-1234-567812345678"
        for resume in (False, True):
            with self.subTest(resume=resume), mock.patch.object(claude, "ensure_auth") as auth:
                factory = providers._factory(
                    "claude", "fake", 4096, cwd=Path.cwd(), effort="low",
                    session_id=session_id, resume=resume)
                primary, child, reset = factory(), factory(), factory()
            auth.assert_called_once_with()
            self.assertEqual(primary.session_id, session_id)
            self.assertEqual(primary.started, resume)
            self.assertEqual(len({model.session_id for model in (primary, child, reset)}), 3)
            for model in (child, reset):
                uuid.UUID(model.session_id)
                self.assertFalse(model.started)
                self.assertEqual(model.model, "fake")
                self.assertEqual(model.effort, "low")
                self.assertEqual(model.cwd, Path.cwd())

    def test_factory_rejects_invalid_session_before_authentication(self):
        for resume in (False, True):
            with self.subTest(resume=resume), mock.patch.object(claude, "ensure_auth") as auth:
                with self.assertRaises(ValueError):
                    providers._factory(
                        "claude", "fake", 4096, cwd=Path.cwd(), effort="high",
                        session_id="invalid-uuid", resume=resume)
                auth.assert_not_called()

    def test_authentication_still_uses_official_cli_login(self):
        with (
            mock.patch.object(claude.shutil, "which", return_value="/bin/claude"),
            mock.patch.object(claude.subprocess, "run") as run,
            mock.patch("builtins.input", return_value=""),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            run.side_effect = [
                SimpleNamespace(returncode=0, stdout='{"loggedIn":false}'),
                SimpleNamespace(returncode=0),
                SimpleNamespace(returncode=0, stdout='{"loggedIn":true}'),
            ]
            claude.ensure_auth()
        status = mock.call(
            ["claude", "auth", "status", "--json"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.assertEqual(run.call_args_list, [
            status,
            mock.call(["claude", "auth", "login", "--claudeai"], check=False),
            status,
        ])


class OpenRouterTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(openrouter.http.client, "HTTPSConnection")
        self.connection = patch.start().return_value
        self.addCleanup(patch.stop)
        self.model = openrouter._OpenRouter("provider/model", "test-key", 64)

    def test_streaming_history_images_and_usage(self):
        self.connection.getresponse.side_effect = [
            stream(chunk("hello "), chunk("world", "stop"),
                   {"choices": [], "usage": {"prompt_tokens": 12}}),
            stream(chunk("done", "stop")),
        ]
        prompt = eko.Message("user", (
            eko.Text("look"), eko.Image("image/png", b"png", "image.png")))
        parts = []
        reply = self.model.complete("system", prompt, parts.append)
        self.assertEqual(parts, ["hello ", "world"])
        self.assertEqual(eko.message_text(reply), "hello world")
        self.assertEqual(self.model.context_used, 12)
        self.model.complete("system", message("next"), lambda _: None)

        request = self.connection.request.call_args
        payload = json.loads(request.args[2])
        self.assertEqual(request.args[:2], ("POST", "/api/v1/chat/completions"))
        self.assertEqual(request.args[3]["Authorization"], "Bearer test-key")
        self.assertEqual(payload["model"], "provider/model")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertTrue(payload["stream"])
        self.assertEqual([turn["role"] for turn in payload["messages"]],
                         ["system", "user", "assistant", "user"])
        self.assertEqual(payload["messages"][1]["content"][-1]["image_url"]["url"],
                         "data:image/png;base64,cG5n")
        self.assertEqual(payload["messages"][2]["content"][0]["text"], "hello world")
        self.assertEqual(self.connection.close.call_count, 2)

    def test_failures_do_not_commit_partial_turns(self):
        responses = [
            stream(chunk("partial", "length")),
            stream(chunk("partial", "content_filter")),
            stream(chunk("partial", "tool_calls")),
            stream(chunk("partial", "stop"), done=False),
            stream(chunk("partial"), {"error": {"message": "failed"}}),
            stream(chunk("", "stop")),
            stream(status=401),
        ]
        for response in responses:
            with self.subTest(response=response):
                self.connection.getresponse.return_value = response
                with self.assertRaises(RuntimeError):
                    self.model.complete("system", message("question"), lambda _: None)
                self.assertEqual(self.model.messages, [])
                self.assertTrue(response.closed)

    def test_interrupt_aborts_connection_and_next_turn_can_complete(self):
        self.connection.getresponse.return_value = stream(chunk("partial", "stop"))
        with self.assertRaises(InterruptedError):
            self.model.complete("system", message("question"),
                                lambda _: self.model.interrupt())
        self.connection.sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.assertEqual(self.model.messages, [])
        self.connection.getresponse.return_value = stream(chunk("done", "stop"))
        reply = self.model.complete("system", message("next"), lambda _: None)
        self.assertEqual(eko.message_text(reply), "done")

    def test_factory_creates_independent_histories(self):
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            factory = providers._factory(
                "openrouter", "provider/model", 64, cwd=Path.cwd(), effort="high")
        first, second = factory(), factory()
        self.connection.getresponse.return_value = stream(chunk("done", "stop"))
        first.complete("system", message("private"), lambda _: None)
        self.assertEqual(second.messages, [])
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                providers._factory(
                    "openrouter", "provider/model", 64, cwd=Path.cwd(), effort="high")


class Tokens(list):
    @property
    def shape(self):
        return (1, len(self))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Tokens(super().__getitem__(key[1]))
        return super().__getitem__(key)


class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        tokenizer = mock.Mock(chat_template="template")
        tokenizer.apply_chat_template.return_value.to.return_value = {
            "input_ids": Tokens([1, 2, 3]), "attention_mask": Tokens([1, 1, 1])}
        tokenizer.decode.return_value = "answer"
        model = mock.Mock(device="cpu")
        model.generation_config.eos_token_id = [0, 9]
        self.runtime = SimpleNamespace(
            tokenizer=tokenizer, model=model, max_tokens=2, lock=threading.Lock())

        class TextStreamer:
            def __init__(self, tokenizer, **kwargs):
                pass

        self.modules = {
            "torch": SimpleNamespace(inference_mode=nullcontext),
            "transformers": SimpleNamespace(
                TextStreamer=TextStreamer, StoppingCriteriaList=list,
                AutoTokenizer=mock.Mock(), AutoModelForCausalLM=mock.Mock()),
        }
        patch = mock.patch.dict(sys.modules, self.modules)
        patch.start()
        self.addCleanup(patch.stop)
        self.model = huggingface._HuggingFace(self.runtime)

        def generate(**kwargs):
            kwargs["streamer"].on_finalized_text("answer", stream_end=True)
            return Tokens([1, 2, 3, 8, 9])

        model.generate.side_effect = generate

    def test_template_streaming_context_and_history(self):
        parts = []
        reply = self.model.complete("system", message("first"), parts.append)
        self.assertEqual(parts, ["answer"])
        self.assertEqual(eko.message_text(reply), "answer")
        self.assertEqual(self.model.context_used, 3)
        self.runtime.tokenizer.decode.assert_called_with([8, 9], skip_special_tokens=True)
        self.model.complete("system", message("second"), lambda _: None)
        call = self.runtime.tokenizer.apply_chat_template.call_args
        self.assertEqual(call.args[0], [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ])
        self.assertTrue(call.kwargs["add_generation_prompt"])
        self.assertTrue(call.kwargs["return_dict"])

    def test_loads_weights_once_for_independent_conversations(self):
        transformers = self.modules["transformers"]
        transformers.AutoTokenizer.from_pretrained.return_value = self.runtime.tokenizer
        transformers.AutoModelForCausalLM.from_pretrained.return_value = self.runtime.model
        factory = providers._factory(
            "huggingface", "/models/chat", 2, cwd=Path.cwd(), effort="high")
        first, second = factory(), factory()
        first.complete("system", message("private"), lambda _: None)
        self.assertIs(first.runtime, second.runtime)
        self.assertEqual(second.messages, [])
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with("/models/chat")
        transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
            "/models/chat", device_map="auto", torch_dtype="auto")

    def test_rejects_images_and_truncated_output(self):
        prompt = eko.Message("user", (eko.Image("image/png", b"png"),))
        with self.assertRaisesRegex(ValueError, "text only"):
            self.model.complete("system", prompt, lambda _: None)
        self.runtime.model.generate.assert_not_called()
        self.runtime.model.generate.side_effect = None
        self.runtime.model.generate.return_value = Tokens([1, 2, 3, 7, 8])
        with self.assertRaisesRegex(RuntimeError, "max-tokens"):
            self.model.complete("system", message("question"), lambda _: None)
        self.assertEqual(self.model.messages, [])
        self.assertFalse(self.runtime.lock.locked())

    def test_interrupt_stops_generation_and_releases_lock(self):
        def generate(**kwargs):
            self.model.interrupt()
            self.assertTrue(kwargs["stopping_criteria"][0](None, None))
            return Tokens([1, 2, 3, 8])

        self.runtime.model.generate.side_effect = generate
        with self.assertRaises(InterruptedError):
            self.model.complete("system", message("question"), lambda _: None)
        self.assertEqual(self.model.messages, [])
        self.assertFalse(self.runtime.lock.locked())

    def test_interrupt_while_waiting_for_shared_model(self):
        self.runtime.lock.acquire()
        errors = []

        def complete():
            try:
                self.model.complete("system", message("question"), lambda _: None)
            except InterruptedError:
                errors.append("interrupted")

        with mock.patch.object(self.model.interrupted, "clear"):
            self.model.interrupt()
            thread = threading.Thread(target=complete, daemon=True)
            thread.start()
            thread.join(1)
        self.runtime.lock.release()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["interrupted"])
        self.runtime.model.generate.assert_not_called()


class HostProviderTests(unittest.TestCase):
    def test_cli_selects_provider(self):
        for provider in ("claude", "openrouter", "huggingface"):
            with (
                mock.patch.object(sys, "argv", [
                    "eko", "--provider", provider, "--model", "model"]),
                mock.patch.object(host, "run") as run,
            ):
                host.main()
            self.assertEqual(run.call_args.kwargs["provider"], provider)
            self.assertEqual(run.call_args.kwargs["model"], "model")

    def test_cli_rejects_missing_model_and_claude_only_options(self):
        options = [[], ["--model", "model", "--resume", "session"],
                   ["--model", "model", "--effort", "high"],
                   ["--model", "model", "--max-tokens", "0"]]
        for provider in ("openrouter", "huggingface"):
            for extra in options:
                with (
                    mock.patch.object(sys, "argv", ["eko", "--provider", provider, *extra]),
                    mock.patch("sys.stderr", new_callable=io.StringIO),
                    self.assertRaises(SystemExit) as error,
                ):
                    host.main()
                self.assertEqual(error.exception.code, 2)

    def test_other_providers_do_not_require_claude_auth(self):
        for provider in (openrouter, huggingface):
            with (
                mock.patch.object(claude, "ensure_auth") as auth,
                mock.patch.object(provider, "_factory", side_effect=ValueError("no model")),
                self.assertRaisesRegex(ValueError, "no model"),
            ):
                host.run(Path.cwd(), None, model="model", effort="high", feral=False,
                         name="Eko", headless=True, sandbox=False,
                         provider=provider.__name__.split(".")[-1])
            auth.assert_not_called()

    def test_model_credentials_stay_in_host_environment(self):
        with (
            mock.patch.dict("os.environ", {
                "OPENROUTER_API_KEY": "router-key", "HF_TOKEN": "hub-token",
            }, clear=True),
            mock.patch.object(providers, "_factory"),
            mock.patch.object(host, "ModelServer"),
            mock.patch.object(host, "AgentProcess", side_effect=RuntimeError("capture")) as agent,
            self.assertRaisesRegex(RuntimeError, "capture"),
        ):
            host.run(Path.cwd(), None, model="model", effort="high", feral=False,
                     name="Eko", headless=True, sandbox=False, provider="openrouter")
        environment = agent.call_args.kwargs["env"]
        self.assertNotIn("OPENROUTER_API_KEY", environment)
        self.assertNotIn("HF_TOKEN", environment)

    def test_socket_protocol_uses_selected_backend(self):
        server, client = socket.socketpair()
        self.addCleanup(client.close)
        backend = mock.Mock(context_used=12)
        backend.complete.return_value = eko.Message("assistant", (eko.Text("answer"),))
        thread = threading.Thread(target=host._model_client,
                                  args=(server, backend), daemon=True)
        thread.start()
        client.sendall((json.dumps({"system": "system"}) + "\n" + json.dumps({
            "message": eko.encode_message(message("question"))}) + "\n").encode())
        client.settimeout(2)
        with client.makefile("rb") as reader:
            reply = json.loads(reader.readline())
        client.shutdown(socket.SHUT_RDWR)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(eko.message_text(eko.decode_message(reply["message"])), "answer")
        self.assertEqual(reply["context_used"], 12)
        backend.close.assert_called_once()

    def test_model_server_continues_after_initialization_failure(self):
        failed, healthy = mock.Mock(), mock.Mock()
        backend = mock.Mock()
        factory = mock.Mock(side_effect=[RuntimeError("initialization failed"), backend])
        with (
            mock.patch.object(host.socket, "socket") as socket_class,
            mock.patch.object(host, "_model_client") as client,
            mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            listener = socket_class.return_value
            listener.accept.side_effect = [(failed, None), (healthy, None), OSError()]
            server = host.ModelServer(Path("/unused.sock"), factory)
            try:
                server._run()
            finally:
                listener.close()
                for thread in server.clients:
                    thread.join(2)
            failed.close.assert_called_once_with()
            client.assert_called_once_with(healthy, backend)
            self.assertIn("initialization failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
