#!/usr/bin/env python3
"""Transcribe one audio file or a folder with Whisper/MMS/CTC models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kartalol_azb_asr.audio import SUPPORTED_AUDIO_EXTENSIONS
from kartalol_azb_asr.inference import ASRTranscriber


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", help="Audio file to transcribe")
    source.add_argument("--folder", help="Folder of audio files to transcribe")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--target-lang", default="azb", help="MMS adapter language")
    parser.add_argument("--output-json", help="Optional JSON output path")
    parser.add_argument("--recursive", action="store_true", help="Recurse into --folder")
    parser.add_argument("--max-new-tokens", type=int, help="Whisper generation limit")
    return parser.parse_args()


def collect_files(args: argparse.Namespace) -> list[Path]:
    if args.audio:
        return [Path(args.audio)]
    folder = Path(args.folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")
    iterator = folder.rglob("*") if args.recursive else folder.glob("*")
    files = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported audio files found in {folder}")
    return files


def main() -> None:
    args = parse_args()
    transcriber = ASRTranscriber.from_pretrained(
        args.model, device=args.device, target_lang=args.target_lang
    )
    generation_kwargs = (
        {"max_new_tokens": args.max_new_tokens} if args.max_new_tokens else None
    )
    results = [
        {
            "audio": str(path),
            "text": transcriber.transcribe_file(path, generation_kwargs=generation_kwargs),
        }
        for path in collect_files(args)
    ]
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(results) == 1:
        print(results[0]["text"])
    else:
        for result in results:
            print(f"{result['audio']}\t{result['text']}")


if __name__ == "__main__":
    main()
