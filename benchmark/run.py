"""Run ECG-MTD dialogues through Eko."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import queue
import tempfile
import time

import eko as core
import host
import providers
from toolbox.leads import describe_leads
from toolbox.registry import Toolbox, add_arguments as add_tool_arguments
from rich.console import Console
from rich.progress import track

from .config import parse_args, save_settings
from .data import append, iter_rows, load_cases, prepare_ecg, response_text

console = Console(highlight=False)
INSTRUCTION = """You are a world-renowned cardiac electrophysiologist. Respond directly to the user's latest question. Keep the answers concise. Interpret findings when relevant.
ECG: record.mat with {leads}.
The file contains waveforms, not findings. {tool_instruction}

{tool_prompts}

End each completed user answer with <done/>.

User: {query}"""


def instruction(tools, leads, query):
    guidance = ("Run the appropriate available tool before making new claims about this ECG. "
                "Execute tools with Python and print their results before answering. "
                "Reuse prior tool results for follow-ups. General questions and farewells do not require tools. "
                "If the available tools cannot answer a question, state the limitation."
                if tools.prompt else "No ECG tools are available. State when ECG findings cannot be determined.")
    return INSTRUCTION.format(leads=leads, query=query, tool_instruction=guidance,
                              tool_prompts=tools.prompt)


def answer(agent, events, prompt, timeout, max_steps):
    """Collect one answer and its execution trace."""
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
        elif event.type == "input":
            text = core.message_text(event.value)
            if text.startswith("[python exit=0]"):
                evidence.append(text)
        elif event.type == "response":
            steps += 1
            text, code = event.value
            text = response_text(text)
            if code is None and text.endswith(core.DONE_MARKER):
                final = text.removesuffix(core.DONE_MARKER).rstrip()
            elif steps >= max_steps:
                return {"status": "error", "error": f"Reached {max_steps} model steps",
                        "trace": trace}
        elif event.type == "state" and event.value == "idle" and steps:
            if not final:
                return {"status": "error", "error": "No completed answer", "trace": trace}
            return {"status": "ok", "response": final, "evidence": evidence,
                    "steps": steps, "trace": trace}


def show_turn(case, turn):
    console.rule(f"ECG {case['case_id']} · Turn {turn['turn'] + 1}/{len(case['user_turns'])}")
    console.print("User:", style="bold")
    console.print(turn["query"], markup=False)
    for event in turn["trace"]:
        if event["type"] == "response":
            console.print("Assistant:", style="bold")
            console.print(response_text(event["text"]).removesuffix(core.DONE_MARKER).rstrip(),
                          markup=False)
        elif event["type"] == "result":
            console.print(f"Python output (exit {event['returncode']}):", style="bold")
            console.print(event["output"].strip() or "(no output)", markup=False)
    if turn["status"] != "ok":
        console.print(f"Error: {turn['error']}", style="red", markup=False)
    console.print()


def run_case(case, ecg_dir, factory, tools, args):
    record = {"case_id": case["case_id"], "ecg_file": case["ecg_file"],
              "status": "ok", "turns": []}
    try:
        # References and result files are outside the agent workspace.
        with tempfile.TemporaryDirectory(prefix="eko-bench-", dir="/tmp") as directory, ExitStack() as resources:
            directory = Path(directory)
            workspace, runtime = directory / "workspace", directory / "runtime"
            workspace.mkdir()
            runtime.mkdir()
            prepare_ecg(ecg_dir, case["ecg_file"], workspace / "record.mat", leads=args.leads)
            sandbox = args.execution == "sandbox"
            tool_env = tools.start(resources, workspace, runtime, sandbox=sandbox)
            server = host.ModelServer(runtime / "model.sock", factory)
            server.start()
            resources.callback(server.close)
            environment = host.agent_environment(runtime, tool_env)
            agent = host.AgentProcess(host.agent_command(
                workspace, runtime, sandbox=sandbox, feral=False, name="Eko",
                python_timeout=args.python_timeout, tool_env=tool_env),
                env={} if sandbox else environment)
            resources.callback(agent.stop)
            events = queue.Queue()
            agent.observer = events.put
            agent.wait_ready()
            leads = describe_leads(args.leads)
            for index, query in enumerate(case["user_turns"]):
                prompt = instruction(tools, leads, query)
                result = answer(agent, events, prompt, args.turn_timeout, args.max_steps)
                turn = {"turn": index, "query": query, **result}
                record["turns"].append(turn)
                if args.max_samples is not None:
                    show_turn(case, turn)
                if result["status"] != "ok":
                    record["status"] = "error"
                    break
    except Exception as error:
        record.update(status="error", error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ecg-dir", required=True, type=Path,
                        help="ECG-Agent's preprocessed MAT directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provider", choices=("sglang", "openrouter", "huggingface"), default="sglang")
    parser.add_argument("--model", required=True)
    parser.add_argument("--sglang-url")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Output tokens per model step (ECG-Agent default: 512)")
    add_tool_arguments(parser)
    parser.add_argument("--execution", choices=("sandbox", "local"), default="sandbox")
    parser.add_argument("--turn-timeout", type=float, default=300,
                        help="Seconds per user turn, including all actions")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Model responses allowed per user turn")
    parser.add_argument("--python-timeout", type=float, default=core.PYTHON_TIMEOUT)
    args = parse_args(parser, "run")
    if args.turn_timeout <= 0 or args.python_timeout <= 0 or args.max_steps <= 0:
        parser.error("--turn-timeout, --python-timeout, and --max-steps must be positive")
    args.ecg_dir = args.ecg_dir.expanduser().resolve()
    try:
        tools = Toolbox(args)
    except ValueError as error:
        parser.error(str(error))
    settings = {key: value for key, value in vars(args).items() if key not in ("config", "output")}
    settings.update(history="generated", temperature=0.0,
                    instruction=instruction(tools, "{leads}", "{query}"))
    save_settings(args.output, settings)
    cases = load_cases(args.dataset, args.split, args.max_samples)
    previous = {row["case_id"]: row.get("status") for row in iter_rows(args.output, missing_ok=True)}
    factory = providers.factory(args.provider, args.model, args.max_tokens, sglang_url=args.sglang_url)
    completed = failed = skipped = 0
    turn_counts = Counter()
    unattempted = 0
    detailed = args.max_samples is not None
    with args.output.open("a") as stream:
        for case in track(cases, description="Generating dialogues", console=console, disable=detailed):
            key = case["case_id"]
            if previous.get(key) == "ok":
                skipped += 1
                if detailed:
                    console.print(f"ECG {key}: skipped (already complete)", markup=False)
                continue
            record = run_case(case, args.ecg_dir, factory, tools, args)
            append(stream, record)
            status = record["status"]
            previous[key] = status
            completed += status == "ok"
            failed += status != "ok"
            turns = record["turns"]
            turn_counts.update(turn["status"] for turn in turns)
            unattempted += len(case["user_turns"]) - len(turns)
            if status == "error":
                error = next((turn["error"] for turn in turns if turn.get("error")),
                             record.get("error", "Unknown error"))
                status += f": {error}"
            if detailed or record["status"] == "error":
                console.print(f"ECG {key}: {status}", markup=False)
    report = {"dialogues_expected": len(cases), "completed": completed, "failed": failed,
              "skipped_complete": skipped, "turns_completed": turn_counts["ok"],
              "turns_failed": turn_counts["error"], "turns_unattempted": unattempted}
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    console.print(f"Dialogues: {completed} completed, {failed} failed, {skipped} skipped (already complete).")
    console.print(f"Turns this run: {turn_counts['ok']} completed, {turn_counts['error']} failed, "
                  f"{unattempted} not attempted.")
    console.print(f"Summary: {path}", markup=False)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
