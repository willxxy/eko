"""Experiment configuration."""

import argparse
import json
from pathlib import Path

from .data import DATASET, LEAD_CONTEXTS


def parse_args(parser, section):
    parser.add_argument("--config", type=Path, help="Experiment JSON; CLI options override it")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--leads", choices=LEAD_CONTEXTS,
                        help="Waveform leads; inferred from the dataset when omitted")
    parser.add_argument("--history", choices=("generated", "ground_truth"), default="generated")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config, _ = probe.parse_known_args()
    if config.config:
        settings = json.loads(config.config.read_text())
        defaults = {key: value for key, value in settings.items() if key not in ("run", "score")}
        defaults.update(settings.get(section, {}))
        parser.set_defaults(**defaults)
        for action in parser._actions:
            if action.dest in defaults:
                action.required = False
    return parser.parse_args()


def save_settings(output, settings):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(output.suffix + ".config.json").write_text(
        json.dumps(settings, indent=2, default=str) + "\n")
