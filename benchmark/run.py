"""Replay ECG-MTD user turns through Eko's normal Python execution loop."""

import argparse
from contextlib import ExitStack
import math
import os
from pathlib import Path
import queue
import shutil
import tempfile
import time

import eko as core
import host
import providers

from .data import DATASET, append, load_cases, prepare_ecg, read_rows

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = """You are an ECG wearable assistant helping a user understand their ECG in record.mat.
Answer the user's latest question directly, briefly, and in natural language.
Speak as a wearable assistant without presenting yourself as a medical professional.

When a question needs new ECG findings, run the appropriate tool and explain its results.
For follow-up questions, use your previous responses and findings; call a tool again only if needed.
If the available tools cannot answer the question, explain the limitation.
Acknowledge thanks and goodbyes naturally.

To measure heart rate (bpm), PR interval, QRS duration, and QTc interval (ms):
from toolbox.measurements.ecg_analysis import ecg_analysis
print(ecg_analysis('record.mat'))
{classification}
Execute tools using Python and print their results before answering.
End each completed user answer with <done/>.

User: {query}"""


def answer(agent, events, prompt, timeout, max_steps):
    """Wait for completion, keeping observable execution separate from the answer."""
    agent.send(prompt)
    deadline = time.monotonic() + timeout
    trace = []
    evidence = []
    final = ""
    steps = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"status": "error", "error": f"User turn exceeded {timeout:g} seconds",
                    "trace": trace}
        try:
            event = events.get(timeout=min(remaining, .2))
        except queue.Empty:
            if agent.proc.poll() is not None:
                return {"status": "error", "error": "Agent exited before completing the user turn",
                        "trace": trace}
            continue
        if event.type in ("delta", "context"):
            continue
        trace.append(core.encode_event(event))
        if event.type == "error":
            return {"status": "error", "error": str(event.value), "trace": trace}
        if event.type == "input":
            text = core.message_text(event.value)
            if text.startswith("[python exit=0]"):
                evidence.append(text)
        if event.type == "response":
            steps += 1
            text, code = event.value
            if code is None and "<done/>" in text:
                final = text.replace("<done/>", "").strip()
            elif steps >= max_steps:
                return {"status": "error", "error": f"Reached {max_steps} model steps",
                        "trace": trace}
        if event.type == "state" and event.value == "idle" and steps:
            if not final:
                return {"status": "error", "error": "No completed answer", "trace": trace}
            return {"status": "ok", "response": final, "evidence": evidence,
                    "steps": steps, "trace": trace}


def run_case(case, ecg_dir, factory, classifier, args):
    record = {"case_id": case["case_id"], "ecg_file": case["ecg_file"], "turns": []}
    # References and result files are outside the agent workspace.
    with tempfile.TemporaryDirectory(prefix="eko-bench-", dir="/tmp") as directory, ExitStack() as resources:
        directory = Path(directory)
        workspace, runtime = directory / "workspace", directory / "runtime"
        workspace.mkdir()
        runtime.mkdir()
        prepare_ecg(ecg_dir, case["ecg_file"], workspace / "record.mat")
        shutil.copytree(ROOT / "toolbox", workspace / "toolbox",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        sandbox = args.execution == "sandbox"
        server = host.ModelServer(runtime / "model.sock", factory)
        server.start()
        resources.callback(server.close)
        environment = os.environ.copy()
        for key in ("OPENROUTER_API_KEY", "SGLANG_API_KEY", "HF_TOKEN", "GOOGLE_API_KEY",
                    "GEMINI_API_KEY", "EKO_CLASSIFIER", "EKO_WORLD"):
            environment.pop(key, None)
        if classifier is not None:
            from toolbox.neural_networks.classification.service import ClassifierServer

            service = ClassifierServer(runtime / "classifier.sock", classifier, workspace,
                                       sandbox=sandbox)
            service.start()
            resources.callback(service.close)
            environment["EKO_CLASSIFIER"] = str(runtime / "classifier.sock")
        agent = host.AgentProcess(host.agent_command(
            workspace, runtime, sandbox=sandbox, feral=False, name="Eko",
            python_timeout=args.python_timeout, classifier=classifier is not None),
            env={} if sandbox else environment)
        resources.callback(agent.stop)
        events = queue.Queue()
        agent.observer = events.put
        try:
            if not agent.ready.wait(10) or agent.proc.poll() is not None:
                raise RuntimeError(agent.startup_error())
            for index, query in enumerate(case["user_turns"]):
                prompt = query
                if index == 0:
                    classification = (
                        "To identify arrhythmias, abnormalities, and other ECG findings:\n"
                        "from toolbox.neural_networks.classification import ecg_classifier\n"
                        "print(ecg_classifier('record.mat'))"
                        if classifier is not None else "No classification checkpoint is loaded.")
                    prompt = INSTRUCTION.format(classification=classification, query=query)
                try:
                    result = answer(agent, events, prompt, args.turn_timeout, args.max_steps)
                except (TimeoutError, RuntimeError) as error:
                    result = {"status": "error", "error": str(error)}
                record["turns"].append({"turn": index, "query": query, **result})
                if result["status"] != "ok":
                    break
            record["status"] = ("ok" if len(record["turns"]) == len(case["user_turns"])
                                and all(t["status"] == "ok" for t in record["turns"]) else "error")
        except (OSError, RuntimeError) as error:
            record.update(status="error", error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, help="Hugging Face dataset ID or local dataset directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--ecg-dir", required=True, type=Path,
                        help="PTB-XL root, records500 directory, or ECG-Agent's prepared MAT directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", choices=("sglang", "openrouter", "huggingface"), default="sglang")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sglang-url")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--classifier-checkpoint", type=Path)
    parser.add_argument("--classifier-device")
    parser.add_argument("--execution", choices=("sandbox", "local"), default="sandbox")
    parser.add_argument("--turn-timeout", type=float, default=300,
                        help="Seconds per user turn, including all actions")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Model responses allowed per user turn")
    parser.add_argument("--python-timeout", type=float, default=core.PYTHON_TIMEOUT)
    args = parser.parse_args()
    if min(args.turn_timeout, args.max_steps, args.python_timeout, args.max_tokens) <= 0:
        parser.error("Timeouts and limits must be positive")
    if not all(math.isfinite(value) for value in (args.turn_timeout, args.python_timeout)):
        parser.error("Timeouts must be finite")
    if args.execution == "sandbox" and not shutil.which("bwrap"):
        parser.error("Sandbox execution requires Linux Bubblewrap; local execution uses --execution local")
    cases = load_cases(args.dataset, args.split, args.max_samples)
    ecg_dir = args.ecg_dir.expanduser().resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous = {r["case_id"]: r for r in read_rows(args.output)} if args.output.exists() else {}
    factory = providers.factory(args.provider, args.model, args.max_tokens, sglang_url=args.sglang_url)
    classifier = None
    if args.classifier_checkpoint:
        from toolbox.neural_networks.classification.model import Classifier

        classifier = Classifier(args.classifier_checkpoint, args.classifier_device)
    failed = 0
    with args.output.open("a") as stream:
        for case in cases:
            key = case["case_id"]
            if previous.get(key, {}).get("status") == "ok":
                continue
            try:
                record = run_case(case, ecg_dir, factory, classifier, args)
            except (OSError, ValueError) as error:
                record = {"case_id": key, "ecg_file": case["ecg_file"],
                          "status": "error", "error": str(error), "turns": []}
            append(stream, record)
            failed += record["status"] != "ok"
            print(f"{key}: {record['status']}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
