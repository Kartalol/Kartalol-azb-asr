#!/usr/bin/env python3
"""Fine-tune any Hugging Face-compatible Whisper checkpoint."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kartalol_azb_asr.data import load_hf_dataset
from _common import apply_config_defaults, load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--dataset", action="append", help="Training dataset alias/ID; repeat for Full")
    parser.add_argument("--dataset-config", action="append", help="Optional config corresponding to --dataset")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--audio-field")
    parser.add_argument("--text-field")
    parser.add_argument("--output-dir", default="outputs/whisper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-audio-duration", type=float, default=30.0)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--generation-num-beams", type=int, default=1)
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--eval-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--language", help="Optional Whisper tokenizer language")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    known, _ = config_parser.parse_known_args(argv)
    parser = build_parser()
    apply_config_defaults(parser, load_yaml(known.config))
    args = parser.parse_args(argv)
    if not args.dataset:
        parser.error("at least one --dataset is required (or set dataset in YAML)")
    if isinstance(args.dataset, str):
        args.dataset = [args.dataset]
    if isinstance(args.dataset_config, str):
        args.dataset_config = [args.dataset_config]
    return args


@dataclass
class WhisperCollator:
    processor: Any
    sampling_rate: int
    max_audio_duration: float

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import numpy as np

        arrays = []
        texts = []
        max_samples = round(self.sampling_rate * self.max_audio_duration)
        for feature in features:
            audio = feature["audio"]
            array = np.asarray(audio["array"], dtype=np.float32)[:max_samples]
            arrays.append(array)
            texts.append(str(feature["text"]))
        batch = self.processor.feature_extractor(
            arrays, sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        label_batch = self.processor.tokenizer(texts, padding=True, return_tensors="pt")
        labels = label_batch.input_ids.masked_fill(label_batch.attention_mask.ne(1), -100)
        if labels.shape[1] and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main() -> None:
    args = parse_args()
    from datasets import Audio, concatenate_datasets
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    set_seed(args.seed)
    configs = args.dataset_config or []
    configs += [None] * (len(args.dataset) - len(configs))
    train_parts = []
    eval_parts = []
    for dataset_name, dataset_config in zip(args.dataset, configs):
        train_parts.append(
            load_hf_dataset(
                dataset_name,
                split=args.train_split,
                config_name=dataset_config,
                for_training=True,
                audio_field=args.audio_field,
                text_field=args.text_field,
            )
        )
        eval_parts.append(
            load_hf_dataset(
                dataset_name,
                split=args.eval_split,
                config_name=dataset_config,
                for_training=True,
                audio_field=args.audio_field,
                text_field=args.text_field,
            )
        )
    train_dataset = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
    eval_dataset = eval_parts[0] if len(eval_parts) == 1 else concatenate_datasets(eval_parts)
    processor_kwargs = {"task": args.task}
    if args.language:
        processor_kwargs["language"] = args.language
    processor = WhisperProcessor.from_pretrained(args.model_id, **processor_kwargs)
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    if "duration" in train_dataset.column_names:
        train_dataset = train_dataset.filter(
            lambda duration: duration is None or duration <= args.max_audio_duration,
            input_columns=["duration"],
        )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.eval_strategy != "no" and args.save_strategy == args.eval_strategy,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=WhisperCollator(processor, sampling_rate, args.max_audio_duration),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
