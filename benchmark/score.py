"""Score saved responses through OpenRouter with ECG-Agent-Play's judge prompts."""

import argparse
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data import DATASET, append, load_cases, read_rows, reference_responses
from .prompts import direct_prompt, faithfulness_prompt, quality_prompt, tool_prompt

JUDGE = "google/gemini-2.5-pro"  # Same judge model as ECG-Agent-Play.


def tasks(cases, responses):
    for case in cases:
        key = case["case_id"]
        record = responses.get(key, {})
        turns = {turn["turn"]: turn for turn in record.get("turns", [])}
        if len(turns) != len(record.get("turns", [])):
            raise ValueError(f"Duplicate user turn for {key}")
        for index, turn in turns.items():
            if not 0 <= index < len(case["user_turns"]) or turn["query"] != case["user_turns"][index]:
                raise ValueError(f"User turn mismatch for {key}:{index}")
            if turn.get("status") == "ok" and not turn.get("response", "").strip():
                raise ValueError(f"Empty completed response for {key}:{index}")
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
    # OpenRouter controls routing and provider safety defaults.
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "stream": False}
    request = Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
            if result.get("error"):
                raise RuntimeError(f"OpenRouter: {result['error']}")
            choices = result.get("choices", [])
            if not choices or choices[0].get("finish_reason") != "stop":
                raise ValueError(f"Judge did not finish: {result}")
            text = choices[0].get("message", {}).get("content")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Judge returned no text")
            return text
        except HTTPError as error:
            if attempt == 2 or error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"Judge HTTP {error.code}: {error.read().decode()}") from error
        except (URLError, TimeoutError):
            if attempt == 2:
                raise
        time.sleep(2 ** attempt)


def parse_scores(text, metric):
    if metric == "faithfulness":
        if text.strip() not in ("0", "1"):
            raise ValueError("Expected a single faithfulness score, 0 or 1")
        return {"faithfulness": int(text.strip())}
    labels = ({"accuracy": "Accuracy", "completeness": "Completeness"} if metric == "response"
              else {"naturalness": "Naturalness Score", "cefr_adherence": "CEFR Adherence Score"})
    scores = {}
    for name, label in labels.items():
        match = re.search(rf"^{label}:\s*([1-5])\s*$", text, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError(f"Missing or invalid {name} score")
        scores[name] = int(match[1])
    return scores


def summarize(rows):
    summary = {}
    for row in rows:
        group = row.get("category", row["metric"])
        entry = summary.setdefault(group, {"total": 0, "status_counts": {}, "means": {}})
        entry["total"] += 1
        status = row["status"]
        entry["status_counts"][status] = entry["status_counts"].get(status, 0) + 1
        for metric, score in row.get("scores", {}).items():
            entry["means"].setdefault(metric, []).append(score)
    for entry in summary.values():
        entry["means"] = {name: sum(values) / len(values) for name, values in entry["means"].items()}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET, help="Hugging Face dataset ID or local dataset directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--judge-model", default=JUDGE)
    parser.add_argument("--dry-run", action="store_true", help="Save judge prompts without API calls")
    args = parser.parse_args()
    cases = load_cases(args.dataset, args.split, args.max_samples)
    responses = {row["case_id"]: row for row in read_rows(args.responses)}
    if responses.keys() - {case["case_id"] for case in cases}:
        parser.error("Response IDs are absent from the selected dataset")
    planned = list(tasks(cases, responses))
    ids = [task["id"] for task in planned]
    if len(set(ids)) != len(ids):
        parser.error("Multiple reference responses for the same user turn")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not args.dry_run and not api_key:
        parser.error("Set OPENROUTER_API_KEY")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = {row["id"]: row for row in read_rows(args.output)} if args.output.exists() else {}
    results = []
    with args.output.open("a") as stream:
        for task in planned:
            previous = existing.get(task["id"])
            if previous and previous["status"] not in ("judge_error", "pending"):
                results.append(previous)
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
                    except (OSError, RuntimeError, ValueError) as error:
                        row.update(status="judge_error", error=str(error))
            append(stream, row)
            results.append(row)
            print(f"{row['id']}: {row['status']}", flush=True)
            if row["status"] == "judge_error":
                # Save progress and stop; rerunning retries this task without repeating successes.
                break
    report = {"dialogues_expected": len(cases),
              "dialogues_completed": sum(responses.get(c["case_id"], {}).get("status") == "ok" for c in cases),
              "tasks_expected": len(planned), "tasks_recorded": len(results),
              "metrics": summarize(results)}
    path = args.output.with_suffix(".summary.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(row["status"] == "judge_error" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
