"""Score responses with ECG-Agent's judge prompts."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
from urllib.request import Request, urlopen

from rich.console import Console
from rich.progress import track

from .config import parse_args, save_settings
from .data import append, lead_context, load_cases, read_rows, reference_responses
from .prompts import direct_prompt, faithfulness_prompt, quality_prompt, tool_prompt

JUDGE = "google/gemini-2.5-pro"
console = Console(highlight=False)


def tasks(cases, responses):
    for case in cases:
        key = case["case_id"]
        record = responses.get(key, {})
        turns = {turn["turn"]: turn for turn in record.get("turns", [])}
        for reference in reference_responses(case["reference_dialogue"]):
            index, category = reference["turn"], reference["category"]
            turn = turns.get(index, {})
            task = {"id": f"{key}:response:{index}", "case_id": key, "metric": "response",
                    "turn": index, "category": category}
            if turn.get("status") != "ok":
                yield {**task, "status": "missing_response"}
                continue
            data = {"response": turn["response"], "context": {"user_query": turn["query"]}}
            if turn.get("evidence"):
                data["tool_output"] = "\n\n".join(turn["evidence"])
            prompt = (direct_prompt(key, data, "eko", reference["response"])
                      if category == "direct_response" else
                      tool_prompt(key, data, "eko", reference["response"], category))
            yield {**task, "prompt": prompt}
        for index in range(len(case["user_turns"])):
            turn = turns.get(index, {})
            task = {"id": f"{key}:faithfulness:{index}", "case_id": key,
                    "metric": "faithfulness", "turn": index}
            if turn.get("status") != "ok":
                yield {**task, "status": "missing_response"}
            elif not turn.get("evidence"):
                yield {**task, "status": "no_evidence"}
            else:
                yield {**task, "prompt": faithfulness_prompt(
                    "\n\n".join(turn["evidence"]), turn["response"])}
        task = {"id": f"{key}:dialogue", "case_id": key, "metric": "dialogue"}
        if any(turns.get(i, {}).get("status") != "ok" for i in range(len(case["user_turns"]))):
            yield {**task, "status": "incomplete_dialogue"}
            continue
        dialogue = "\n".join(
            f"User: {query}\nAssistant: {turns[i]['response']}"
            for i, query in enumerate(case["user_turns"]))
        yield {**task, "prompt": quality_prompt(dialogue, case["cefr_level"])}


def judge(prompt, model, api_key):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": False}
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=300) as response:
        choice = json.load(response)["choices"][0]
    return choice["message"]["content"] if choice["finish_reason"] == "stop" else ""


def parse_scores(text, metric):
    if metric == "faithfulness":
        return {"faithfulness": int(re.fullmatch("[01]", text.strip())[0])}
    labels = ({"accuracy": "Accuracy", "completeness": "Completeness"} if metric == "response"
              else {"naturalness": "Naturalness Score", "cefr_adherence": "CEFR Adherence Score"})
    scores = {}
    for name, label in labels.items():
        match = re.search(rf"^{label}:\s*([1-5])\s*$", text, re.MULTILINE | re.IGNORECASE)
        scores[name] = int(match[1])
    return scores


def summarize(rows):
    summary = {}
    for row in rows:
        group = row.get("category", row["metric"])
        entry = summary.setdefault(group, {"total": 0, "scored": 0, "status_counts": {}, "means": {}})
        entry["total"] += 1
        status = row["status"]
        entry["scored"] += status == "ok"
        entry["status_counts"][status] = entry["status_counts"].get(status, 0) + 1
        for metric, score in row.get("scores", {}).items():
            entry["means"].setdefault(metric, []).append(score)
    for entry in summary.values():
        entry["means"] = {name: sum(values) / len(values) for name, values in entry["means"].items()}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-model", default=JUDGE)
    parser.add_argument("--dry-run", action="store_true", help="Save judge prompts without API calls")
    args = parse_args(parser, "score")
    args.leads = lead_context(args.dataset, args.leads)
    response_config = args.responses.with_suffix(args.responses.suffix + ".config.json")
    if response_config.exists():
        settings = json.loads(response_config.read_text())
        changed = [key for key in ("dataset", "split", "leads", "history")
                   if settings.get(key) != getattr(args, key)]
        if changed:
            parser.error(f"Response settings differ: {', '.join(changed)}")
    save_settings(args.output, {key: value for key, value in vars(args).items()
                                if key not in ("config", "output", "dry_run")})
    cases = load_cases(args.dataset, args.split, args.max_samples, args.leads)
    responses = {row["case_id"]: row for row in read_rows(args.responses)}
    planned = list(tasks(cases, responses))
    api_key = None if args.dry_run else os.environ["OPENROUTER_API_KEY"]
    existing = {row["id"]: row for row in read_rows(args.output)} if args.output.exists() else {}
    results = []
    resumed = 0
    detailed = args.max_samples is not None
    with args.output.open("a") as stream:
        for task in track(planned, description="Scoring responses", console=console, disable=detailed):
            previous = existing.get(task["id"])
            if (previous and previous["status"] not in ("judge_error", "pending")
                    and all(previous.get(key) == value for key, value in task.items())
                    and ("prompt" in previous) == ("prompt" in task)):
                results.append(previous)
                resumed += 1
                continue
            row = dict(task)
            if "status" not in row:
                if args.dry_run:
                    row["status"] = "pending"
                else:
                    row["raw_response"] = ""
                    try:
                        row["raw_response"] = judge(row["prompt"], args.judge_model, api_key)
                        row["scores"] = parse_scores(row["raw_response"], row["metric"])
                        row["status"] = "ok"
                    except Exception as error:
                        row.update(status="judge_error", error=str(error))
            append(stream, row)
            results.append(row)
            if detailed or row["status"] == "judge_error":
                scores = ", ".join(f"{key}={value}" for key, value in row.get("scores", {}).items())
                detail = row.get("error") or scores
                console.print(f"{row['id']}: {row['status']}" + (f" ({detail})" if detail else ""),
                              markup=False)
            if row["status"] == "judge_error":
                break
    counts = Counter(row["status"] for row in results)
    skipped = {status: count for status, count in counts.items()
               if status not in ("ok", "pending", "judge_error")}
    report = {"dataset": args.dataset, "split": args.split, "leads": args.leads,
              "history": args.history, "judge_model": args.judge_model,
              "dialogues_expected": len(cases),
              "dialogues_completed": sum(responses.get(c["case_id"], {}).get("status") == "ok" for c in cases),
              "tasks_expected": len(planned), "tasks_recorded": len(results),
              "tasks_resumed": resumed, "tasks_unprocessed": len(planned) - len(results),
              "status_counts": dict(counts), "skipped": skipped,
              "metrics": summarize(results)}
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    console.print(f"Tasks: {len(results)}/{len(planned)} recorded, {resumed} reused, "
                  f"{report['tasks_unprocessed']} not processed.")
    console.print(f"Scores: {counts['ok']} completed, {counts['judge_error']} failed, "
                  f"{counts['pending']} pending, {sum(skipped.values())} skipped.")
    for status, count in sorted(skipped.items()):
        console.print(f"  {status}: {count}")
    for group, entry in report["metrics"].items():
        means = ", ".join(f"{name}={value:.3f}" for name, value in entry["means"].items())
        console.print(f"{group}: {entry['scored']}/{entry['total']} scored" +
                      (f"; {means}" if means else ""))
    console.print(f"Summary: {path}", markup=False)
    if any(row["status"] == "judge_error" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
