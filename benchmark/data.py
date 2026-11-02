"""ECG-MTD dialogues and PTB-XL waveforms."""

import json
import os
from pathlib import Path

DATASET = "gustmd0121/12-lead-ecg-mtd-dataset"
LEAD_CONTEXTS = ("12-lead", "single-lead-I", "single-lead-II")


def lead_context(dataset, requested=None):
    leads = requested or Path(dataset).name.split("-ecg-mtd-dataset")[0]
    if leads not in LEAD_CONTEXTS:
        raise ValueError(f"Unknown lead context {leads!r}; specify --leads")
    return leads


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def response_text(text):
    """Remove Qwen reasoning; its opening tag may be prefilled by the template."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        raise ValueError("Response contains an unclosed reasoning block")
    return text.strip()


def reference_responses(dialogue):
    """ECG-Agent response categories by user-turn position."""
    responses = []
    user_index = -1
    query = ""
    for index, turn in enumerate(dialogue):
        if turn["role"] == "user":
            user_index += 1
            query = turn["content"]
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
                              "query": query, "response": turn["content"]})
    return responses


def load_references(dataset, split):
    """Index judge references by ECG ID and exact user question."""
    from datasets import load_dataset

    references = {}
    for row in load_dataset(dataset, split=split):
        key = str(json.loads(row["metadata"])["ecg_id"])
        if key in references:
            raise ValueError(f"Duplicate reference ECG ID: {key}")
        queries = {}
        for reference in reference_responses(json.loads(row["dialogue"])):
            query = reference["query"]
            if query in queries:
                raise ValueError(f"Duplicate reference question for ECG {key}: {query!r}")
            queries[query] = reference
        references[key] = queries
    return references


def load_cases(dataset=DATASET, split="test", max_samples=None, leads=None):
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split)
    if max_samples is not None:
        rows = rows.select(range(min(max_samples, len(rows))))
    leads = lead_context(dataset, leads)
    cases = []
    for row in rows:
        metadata = json.loads(row["metadata"])
        dialogue = json.loads(row["dialogue"])
        cases.append({"case_id": str(metadata["ecg_id"]), "leads": leads,
                      "ecg_file": metadata["ecg_lookup_filename_used"],
                      "cefr_level": metadata["cefr_level"],
                      "user_turns": [turn["content"] for turn in dialogue if turn["role"] == "user"],
                      "reference_dialogue": dialogue})
    return cases


def prepare_ecg(ecg_dir, ecg_file, destination):
    """Copy ECG-Agent's 12-lead tool inputs without training labels."""
    import numpy as np
    from scipy.io import loadmat, savemat
    ecg_dir = Path(ecg_dir).resolve()
    # ECG-Agent preprocessing writes HR25.mat; the dataset names HR00025.mat.
    ecg_id = int(Path(ecg_file).stem.removeprefix("HR"))
    source = (ecg_dir / f"HR{ecg_id}.mat").resolve()
    source.relative_to(ecg_dir)
    data = loadmat(str(source))
    signal = np.asarray(data["feats"])
    rate = data.get("curr_sample_rate")

    rate = float(np.asarray(500 if rate is None else rate).item())
    if rate != 500:
        raise ValueError("ECG-Agent's benchmark tools require 500 Hz ECGs")
    if signal.ndim != 2 or signal.shape[0] != 12 or not signal.shape[1]:
        raise ValueError("Expected feats with shape (12, samples) in ECG-Agent lead order")
    if not np.isfinite(signal).all():
        raise ValueError("ECG contains nonfinite values")
    savemat(destination, {"feats": signal, "curr_sample_rate": int(rate)})
