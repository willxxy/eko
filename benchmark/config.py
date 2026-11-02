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
                        help="Dialogue lead context; tools always receive all 12 leads")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config, _ = probe.parse_known_args()
    if config.config:
        settings = json.loads(config.config.read_text())
        defaults = {key: value for key, value in settings.items() if key not in ("run", "score")}
        defaults.update(settings.get(section, {}))
        unknown = defaults.keys() - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"Unknown config options: {', '.join(sorted(unknown))}")
        parser.set_defaults(**defaults)
        for action in parser._actions:
            if defaults.get(action.dest) is not None:
                action.required = False
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    if args.leads is not None and args.leads not in LEAD_CONTEXTS:
        parser.error(f"--leads must be one of {', '.join(LEAD_CONTEXTS)}")
    return args


def save_settings(output, settings):
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.with_suffix(output.suffix + ".config.json")
    settings = json.loads(json.dumps(settings, default=str))
    if output.exists() and output.stat().st_size and path.exists():
        previous = json.loads(path.read_text())
        changed = sorted(key for key in previous.keys() | settings.keys()
                         if key != "max_samples" and previous.get(key) != settings.get(key))
        if changed:
            raise ValueError(f"Existing output uses different settings: {', '.join(changed)}. "
                             "Choose a separate output file.")
    path.write_text(json.dumps(settings, indent=2) + "\n")
