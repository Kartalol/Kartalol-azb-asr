#!/usr/bin/env python3
"""Evaluate supplied checkpoints on External, Community, and GoldSet."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
from _common import load_yaml, write_json

TEST_SETS = ("external", "community", "goldset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="YAML checkpoint manifest")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalization", choices=("none", "published", "canonical"), default="none")
    parser.add_argument("--max-samples", type=int, help="Debug-only limit per test set")
    return parser.parse_args()


def metric_cell(metrics: dict[str, Any]) -> str:
    return f"{metrics['wer']:.1f} / {metrics['cer']:.1f} / {metrics['dir']:.2f}"


def main() -> None:
    args = parse_args()
    manifest = load_yaml(args.manifest)
    models = manifest.get("models", [])
    datasets = manifest.get("datasets", {})
    if not models:
        raise ValueError("Manifest must contain a non-empty 'models' list")
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for model in models:
        row: dict[str, Any] = {
            "Model": model["name"],
            "Training Setup": model["training_setup"],
        }
        for test_set in TEST_SETS:
            spec = datasets.get(test_set, {"dataset": test_set, "split": "test"})
            result_dir = output_dir / model["slug"] / test_set
            command = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--model",
                model["checkpoint"],
                "--dataset",
                spec.get("dataset", test_set),
                "--split",
                spec.get("split", "test"),
                "--device",
                args.device,
                "--normalization",
                args.normalization,
                "--output-dir",
                str(result_dir),
            ]
            if spec.get("config"):
                command += ["--dataset-config", spec["config"]]
            if args.max_samples is not None:
                command += ["--max-samples", str(args.max_samples)]
            subprocess.run(command, check=True)
            metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
            row[test_set.title() if test_set != "goldset" else "GoldSet"] = metric_cell(metrics)
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["Model", "Training Setup", "External", "Community", "GoldSet"]
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "benchmark.json", rows)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for row in rows)
    table = "\n".join(lines) + "\n"
    (output_dir / "benchmark.md").write_text(table, encoding="utf-8")
    print(table, end="")


if __name__ == "__main__":
    main()
