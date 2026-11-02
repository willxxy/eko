"""Freeze ECG-MTD rows into a local benchmark; shared JSONL helpers."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def read_rows(path):
    path = Path(path)
    with path.open() as stream:
        if path.suffix == ".json":
            rows = json.load(stream)
        else:
            rows = [json.loads(line) for line in stream if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected a JSON array or JSONL objects: {path}")
    return rows


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def check_run(output, config):
    """Resume only when inputs, code, and settings still match."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = output.with_suffix(output.suffix + ".meta.json")
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError(f"Run settings changed; use a new output: {output}")
    else:
        if output.exists():
            raise ValueError(f"Missing run metadata for {output}; use a new output")
        manifest.write_text(json.dumps(config, indent=2) + "\n")


def user_turns(dialogue):
    return [turn["content"] for turn in dialogue if turn["role"] == "user"]


def reference_responses(dialogue):
    """ECG-Agent-Play categories, keyed by user-turn position to retain repeats."""
    responses = []
    user_index = -1
    for index, turn in enumerate(dialogue):
        if turn["role"] == "user":
            user_index += 1
        if turn["role"] != "assistant" or turn.get("action") != "response":
            continue
        previous = dialogue[index - 1] if index else {}
        action = previous.get("action")
        category = None
        if action == "call_classification_tool":
            category = "post_classification"
        elif action == "call_measurement_tool":
            category = "post_measurement"
        elif previous.get("role") == "user" or action == "response_followup":
            category = "direct_response"
        if category and user_index >= 0:
            responses.append({"turn": user_index, "category": category,
                              "response": turn["content"]})
    return responses


def load_cases(path):
    cases = read_rows(path)
    validate_cases(cases)
    return cases


def validate_cases(cases):
    seen = set()
    for case in cases:
        key = case["case_id"]
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError(f"Missing or duplicate case ID: {key}")
        seen.add(key)
        path = Path(case["ecg_file"])
        if path.is_absolute() or ".." in path.parts or path.suffix != ".mat":
            raise ValueError(f"Expected a relative .mat path: {path}")
        if case["cefr_level"] not in ("A", "B", "C", "A1", "A2", "B1", "B2", "C1", "C2"):
            raise ValueError(f"Missing or invalid CEFR level for {key}")
        queries = case["user_turns"]
        if not queries or not all(isinstance(q, str) and q.strip() for q in queries):
            raise ValueError(f"Missing user turns for {key}")
        if queries != user_turns(case["reference_dialogue"]):
            raise ValueError(f"Reference user turns differ for {key}")
        responses = reference_responses(case["reference_dialogue"])
        if len({r["turn"] for r in responses}) != len(responses):
            raise ValueError(f"Multiple reference responses for one user turn: {key}")
        if any(not isinstance(r["response"], str) or not r["response"].strip() for r in responses):
            raise ValueError(f"Empty reference response for {key}")
    if not cases:
        raise ValueError("Empty benchmark")


def load_source(name, split, revision):
    if Path(name).is_file():
        return read_rows(name), {"file": str(Path(name).resolve()), "sha256": digest(name)}
    if not revision:
        raise ValueError("Remote datasets require --revision (use a commit SHA)")
    from datasets import load_dataset

    dataset = load_dataset(name, split=split, revision=revision)
    return dataset, {"dataset": name, "split": split, "revision": revision,
                     "fingerprint": dataset._fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="HF dataset ID or local JSON/JSONL export")
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision")
    parser.add_argument("--reference", help="Optional separate reference dataset/export")
    parser.add_argument("--reference-split", default="train")
    parser.add_argument("--reference-revision")
    parser.add_argument("--ids", type=Path, help="ECG IDs, one per line, from a comparison cohort")
    parser.add_argument("--limit", type=int, help="First N selected dialogues in source order")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    rows, source = load_source(args.dataset, args.split, args.revision)
    references = {}
    if args.reference:
        reference_rows, source["reference"] = load_source(
            args.reference, args.reference_split, args.reference_revision)
        for row in reference_rows:
            key = str(decode(row["metadata"])["ecg_id"])
            if key in references:
                raise ValueError(f"Duplicate reference ECG ID: {key}")
            references[key] = decode(row["dialogue"])
    ids = set(args.ids.read_text().split()) if args.ids else None
    cases = []
    for row in rows:
        metadata = decode(row["metadata"])
        key = str(metadata["ecg_id"])
        if ids is not None and key not in ids:
            continue
        dialogue = decode(row["dialogue"])
        reference = references[key] if args.reference else dialogue
        files = decode(row["ecg_files"])
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError(f"Expected exactly one ECG file for {key}")
        cases.append({"case_id": key, "ecg_id": key, "ecg_file": files[0],
                      "cefr_level": metadata["cefr_level"], "user_turns": user_turns(dialogue),
                      "reference_dialogue": reference, "source": source})
    if ids is not None and ids - {c["case_id"] for c in cases}:
        raise ValueError("Requested cohort IDs are absent from the dataset")
    if args.limit:
        cases = cases[:args.limit]
    validate_cases(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for case in cases:
            append(stream, case)
    print(f"Saved {len(cases)} dialogues to {args.output}")


if __name__ == "__main__":
    main()
