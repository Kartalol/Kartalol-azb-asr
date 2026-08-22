#!/usr/bin/env python3
"""Evaluate a Hugging Face Whisper or CTC model on an ASR dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kartalol_azb_asr.data import load_hf_dataset
from kartalol_azb_asr.inference import ASRTranscriber
from kartalol_azb_asr.metrics import compute_asr_metrics
from kartalol_azb_asr.normalization import normalize_azb
from _common import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="community, external, goldset, or HF ID")
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-field")
    parser.add_argument("--text-field")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-lang", default="azb")
    parser.add_argument(
        "--normalization",
        choices=("none", "published", "canonical"),
        default="none",
        help="Official scores use 'none'; normalized evaluation must be requested explicitly",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--max-samples", type=int, help="Debug-only sample limit")
    return parser.parse_args(argv)


def audio_array(value: Any) -> tuple[Any, int]:
    if not isinstance(value, dict) or "array" not in value:
        raise ValueError("Dataset audio must decode to a {'array', 'sampling_rate'} mapping")
    return value["array"], int(value["sampling_rate"])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_hf_dataset(
        args.dataset,
        split=args.split,
        config_name=args.dataset_config,
        audio_field=args.audio_field,
        text_field=args.text_field,
    )
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    transcriber = ASRTranscriber.from_pretrained(
        args.model, device=args.device, target_lang=args.target_lang
    )
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(dataset):
        samples, sampling_rate = audio_array(item["audio"])
        if sampling_rate != transcriber.sampling_rate:
            from kartalol_azb_asr.audio import prepare_audio_array

            samples = prepare_audio_array(samples, sampling_rate, transcriber.sampling_rate)
        reference_raw = str(item["text"])
        prediction_raw = transcriber.transcribe_array(samples)
        reference = normalize_azb(reference_raw, profile=args.normalization)
        prediction = normalize_azb(prediction_raw, profile=args.normalization)
        utterance = compute_asr_metrics([reference], [prediction])
        rows.append(
            {
                "index": index,
                "reference": reference,
                "prediction": prediction,
                "reference_raw": reference_raw,
                "prediction_raw": prediction_raw,
                "wer": utterance["wer"],
                "cer": utterance["cer"],
                "dir": utterance["dir"],
            }
        )
    metrics = compute_asr_metrics(
        [row["reference"] for row in rows], [row["prediction"] for row in rows]
    )
    metrics.update(
        {
            "model": args.model,
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "normalization": args.normalization,
            "wer_cer_unit": "percent",
            "dir_unit": "raw_ratio",
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "metrics.json", metrics)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["index", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(rows)
    return metrics


def main() -> None:
    metrics = evaluate(parse_args())
    print(f"WER: {metrics['wer']:.2f}%")
    print(f"CER: {metrics['cer']:.2f}%")
    print(f"DIR: {metrics['dir']:.4g}")


if __name__ == "__main__":
    main()
