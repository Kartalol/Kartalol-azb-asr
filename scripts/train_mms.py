#!/usr/bin/env python3
"""Fine-tune the AZB adapter of facebook/mms-1b-all."""

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
    parser.add_argument("--config")
    parser.add_argument("--model-id", default="facebook/mms-1b-all")
    parser.add_argument("--target-lang", default="azb")
    parser.add_argument("--dataset", default="community")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--audio-field")
    parser.add_argument("--text-field")
    parser.add_argument("--output-dir", default="outputs/mms-community")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-audio-duration", type=float, default=30.0)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(argv)
    parser = build_parser()
    apply_config_defaults(parser, load_yaml(known.config))
    return parser.parse_args(argv)


@dataclass
class CTCCollator:
    processor: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [{"input_values": feature["input_values"]} for feature in features]
        labels = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(inputs, padding=True, return_tensors="pt")
        label_batch = self.processor.tokenizer.pad(labels, padding=True, return_tensors="pt")
        batch["labels"] = label_batch.input_ids.masked_fill(
            label_batch.attention_mask.ne(1), -100
        )
        return batch


def main() -> None:
    args = parse_args()
    import numpy as np
    from datasets import Audio
    from transformers import AutoProcessor, Trainer, TrainingArguments, Wav2Vec2ForCTC, set_seed

    set_seed(args.seed)
    train_dataset = load_hf_dataset(
        args.dataset,
        split=args.train_split,
        config_name=args.dataset_config,
        for_training=True,
        audio_field=args.audio_field,
        text_field=args.text_field,
    )
    eval_dataset = load_hf_dataset(
        args.dataset,
        split=args.eval_split,
        config_name=args.dataset_config,
        for_training=True,
        audio_field=args.audio_field,
        text_field=args.text_field,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.set_target_lang(args.target_lang)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)
    model.load_adapter(args.target_lang)
    model.config.ctc_zero_infinity = True
    if args.adapter_only:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for name, parameter in model.named_parameters():
            if "adapter" in name or "lm_head" in name:
                parameter.requires_grad = True
    model.gradient_checkpointing_enable()
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    if "duration" in train_dataset.column_names:
        train_dataset = train_dataset.filter(
            lambda duration: duration is None or duration <= args.max_audio_duration,
            input_columns=["duration"],
        )

    def prepare(row: dict[str, Any]) -> dict[str, Any]:
        audio = row["audio"]
        values = processor(
            np.asarray(audio["array"]), sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        labels = processor.tokenizer(row["text"]).input_ids
        return {"input_values": values, "labels": labels}

    train_dataset = train_dataset.map(prepare, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare, remove_columns=eval_dataset.column_names)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        data_seed=args.seed,
        group_by_length=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CTCCollator(processor),
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
