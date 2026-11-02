"""ECG-MTD dialogues and PTB-XL waveforms."""

import json
import os
from pathlib import Path

DATASET = "gustmd0121/12-lead-ecg-mtd-dataset"
LEAD_CONTEXTS = ("12-lead", "single-lead-I", "single-lead-II")


def lead_context(dataset, requested=None):
    return requested or Path(dataset).name.split("-ecg-mtd-dataset")[0]


def reference_history(dialogue, user_index):
    """Reference turns before the selected user turn."""
    history = []
    position = -1
    for turn in dialogue:
        if turn["role"] == "user":
            position += 1
        if position >= user_index:
            break
        entry = {key: turn[key] for key in ("role", "action", "content", "tool_output")
                 if key in turn}
        history.append(entry)
    return history


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def reference_responses(dialogue):
    """ECG-Agent response categories by user-turn position."""
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


def load_cases(dataset=DATASET, split="test", max_samples=None, leads=None):
    from datasets import load_dataset

    rows = load_dataset(dataset, split=f"{split}[:{max_samples}]" if max_samples else split)
    leads = lead_context(dataset, leads)
    cases = []
    for row in rows:
        metadata = json.loads(row["metadata"])
        dialogue = json.loads(row["dialogue"])
        cases.append({"case_id": str(metadata["ecg_id"]), "leads": leads,
                      "ecg_file": json.loads(row["ecg_files"])[0],
                      "cefr_level": metadata["cefr_level"],
                      "user_turns": [turn["content"] for turn in dialogue if turn["role"] == "user"],
                      "reference_dialogue": dialogue})
    return cases


def prepare_ecg(ecg_dir, ecg_file, destination, leads="12-lead"):
    """Write the selected leads from a MAT or PTB-XL record."""
    import numpy as np
    from scipy.io import loadmat, savemat
    print(ecg_dir)
    print(ecg_file)

    ecg_dir = Path(ecg_dir).resolve()
    print(ecg_dir)
    source = (ecg_dir / ecg_file).resolve()
    source.relative_to(ecg_dir)
    print(source)
    if source.is_file():
        data = loadmat(source)
        signal = np.asarray(data["feats"])
        rate = data.get("curr_sample_rate")
    else:
        import wfdb

        ecg_id = int(Path(ecg_file).stem.removeprefix("HR"))
        folder = f"{ecg_id // 1000 * 1000:05d}"
        records = ecg_dir / "records500" if (ecg_dir / "records500").is_dir() else ecg_dir
        record = records / folder / f"{ecg_id:05d}_hr"
        signal, metadata = wfdb.rdsamp(str(record))
        signal = signal.T
        rate = metadata["fs"]

    rate = int(np.asarray(500 if rate is None else rate).item())
    if rate != 500:
        raise ValueError("ECG-Agent's benchmark tools require 500 Hz ECGs")
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError("Expected feats with shape (12, samples) in ECG-Agent lead order")
    measurement_lead = 0 if leads == "single-lead-I" else 1
    if leads != "12-lead":
        # Classifiers expect 12 channels; unavailable leads stay zero.
        masked = np.zeros_like(signal)
        masked[measurement_lead] = signal[measurement_lead]
        signal = masked
    # Exclude training labels.
    prepared = {"feats": signal, "measurement_lead": measurement_lead, "lead_context": leads}
    if rate is not None:
        prepared["curr_sample_rate"] = rate
    savemat(destination, prepared)
