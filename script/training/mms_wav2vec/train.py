# © 2026 Kartal Ol NGO. Confidential and proprietary.
# Internal use only. Unauthorized use or distribution is prohibited.

"""
Train MMS (facebook/mms-1b-all) on a JSON manifest:

manifest.json:
{
  "data": [
    {"audio_filepath": "m2_ID565_7912_book3.wav", "duration": 4.608, "text": "..."},
    ...
  ]
}

Audio files live under: dataset/   (so full path = dataset/<audio_filepath>)

This script:
- loads JSON
- builds HF datasets
- loads MMS + sets target_lang + loads adapter
- resamples to 16k, dynamic padding
- trains (by default: only adapter + lm_head)
- evaluates WER/CER

pip install -U transformers datasets accelerate evaluate jiwer torchaudio soundfile
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torchaudio
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate


# -----------------------
# Metrics
# -----------------------
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(processor: AutoProcessor):
    def _compute(pred):
        pred_logits = pred.predictions
        pred_ids = torch.from_numpy(pred_logits).argmax(dim=-1)
        pred_str = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return _compute


# -----------------------
# JSON manifest -> Dataset
# -----------------------
def load_json_manifest(json_path: str, audio_root: str) -> Dataset:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = obj.get("data", [])
    paths, texts, durs = [], [], []
    for r in rows:
        rel = r["audio_filepath"]
        p = rel if os.path.isabs(rel) else os.path.join(audio_root, rel)
        paths.append(p)
        texts.append(r.get("text", ""))
        durs.append(float(r.get("duration", 0.0)))

    return Dataset.from_dict({"path": paths, "text": texts, "duration": durs})

# -----------------------
# Audio loader
# -----------------------
def load_audio_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)  # (channels, time)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    wav = wav.squeeze(0)  # (time,)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav

# -----------------------
# Preprocess
# -----------------------
def prepare_dataset(ds: Dataset, processor: AutoProcessor) -> Dataset:
    def _map(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = load_audio_16k(batch["path"])
        inputs = processor(audio.numpy(), sampling_rate=16000)
        with processor.as_target_processor():
            labels = processor(batch["text"]).input_ids
        return {
            "input_values": inputs["input_values"][0],  # list[float]
            "labels": labels,
        }

    return ds.map(_map, remove_columns=ds.column_names)


# -----------------------
# Collator (dynamic padding + CTC labels)
# -----------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, default="/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/KartalOl_general_voices_0_0_1_training.json")
    ap.add_argument("--audio_root", type=str, default="/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/KartalOl_azb_voices_0_0_1")
    ap.add_argument("--output_dir", type=str, default="./mms_azb_ft")
    ap.add_argument("--model_id", type=str, default="facebook/mms-1b-all")
    ap.add_argument("--target_lang", type=str, default="azb")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--num_train_epochs", type=int, default=10)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--train_adapter_only", action="store_true", default=True)
    args = ap.parse_args()

    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id)

    # MMS adapter setup
    processor.tokenizer.set_target_lang(args.target_lang)
    model.load_adapter(args.target_lang)

    # CTC stability
    model.config.ctc_zero_infinity = True

    # Train only adapter + lm_head (recommended)
    if args.train_adapter_only:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if "adapter" in n or "lm_head" in n:
                p.requires_grad = True

    model.gradient_checkpointing_enable()

    full_ds = load_json_manifest(args.train_json, args.audio_root)
    full_ds = full_ds.shuffle(seed=args.seed)
    split = full_ds.train_test_split(test_size=0.01, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    # (Optional) quick sanity checks: missing files
    def _check_files(ds: Dataset, name: str):
        missing = [p for p in ds["path"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{name}: {len(missing)} audio files missing. Example: {missing[0]}"
            )

    _check_files(train_ds, "train")
    _check_files(eval_ds, "eval")

    train_ds = prepare_dataset(train_ds, processor)
    eval_ds = prepare_dataset(eval_ds, processor)

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=4,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics(processor),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()

"""
python train.py \
    --
  --output_dir ./mms_azb_ft \
  --target_lang azb \
  --fp16 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_train_epochs 10 \
  --train_adapter_only

"""