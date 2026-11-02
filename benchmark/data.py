"""Load ECG-MTD dialogues and prepare ECG-Agent's local ECG files."""

import json
import os
from pathlib import Path
import shutil

DATASET = "gustmd0121/12-lead-ecg-mtd-dataset"


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


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


def load_cases(dataset=DATASET, split="test", max_samples=None):
    from datasets import load_dataset

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    rows = load_dataset(dataset, split=f"{split}[:{max_samples}]" if max_samples else split)
    cases = []
    for row in rows:
        metadata = json.loads(row["metadata"])
        dialogue = json.loads(row["dialogue"])
        cases.append({"case_id": str(metadata["ecg_id"]),
                      "ecg_file": json.loads(row["ecg_files"])[0],
                      "cefr_level": metadata["cefr_level"],
                      "user_turns": [turn["content"] for turn in dialogue if turn["role"] == "user"],
                      "reference_dialogue": dialogue})
    return cases


def prepare_ecg(ecg_dir, ecg_file, destination):
    """Reuse ECG-Agent MAT files, or prepare a raw PTB-XL record the same way."""
    source = (ecg_dir / ecg_file).resolve()
    source.relative_to(ecg_dir)
    if source.is_file():
        shutil.copyfile(source, destination)
        return

    import numpy as np
    from scipy.io import savemat
    import wfdb

    # PTB-XL's records500 layout; ECG-Agent names its prepared files HR<id>.mat.
    ecg_id = int(Path(ecg_file).stem.removeprefix("HR"))
    folder = f"{ecg_id // 1000 * 1000:05d}"
    records = ecg_dir / "records500" if (ecg_dir / "records500").is_dir() else ecg_dir
    record = records / folder / f"{ecg_id:05d}_hr"
    signal, metadata = wfdb.rdsamp(str(record))
    if metadata["fs"] != 500 or signal.shape[1] != 12 or np.isnan(signal).any():
        raise ValueError(f"Expected a 500 Hz, 12-lead ECG without NaNs: {record}")
    # Match src/preprocess/preprocess_ptbxl.py: transpose, keep all leads, no scaling.
    savemat(destination, {"feats": signal.T, "curr_sample_rate": metadata["fs"]})
