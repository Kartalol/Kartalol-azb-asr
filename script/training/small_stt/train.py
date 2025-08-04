
import os
import pandas as pd
import json
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# Set your audio directory and JSON path
audio_dir = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/kartal_general_voices/"
json_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/training_20250717.json"

# Load JSON data
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

# Convert to DataFrame and prepare columns
df = pd.DataFrame(data)
df["audio"] = df["audio_filepath"].apply(lambda x: os.path.join(audio_dir, x))

df.rename(columns={"text": "sentence"}, inplace=True)

# Create Hugging Face Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
# Load Whisper model and processor
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Dynamic data collator
import numpy as np

class WhisperDataCollator:
    def __init__(self, processor, sampling_rate=16000, max_duration=30.0):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.max_samples = int(sampling_rate * max_duration)

    def __call__(self, features):
        padded_audios = []
        texts = []

        for f in features:
            array = f["audio"]["array"]
            text = f["sentence"]

            if not isinstance(text, str) or not text.strip():
                text = "<unk>"  # fallback if empty or invalid

            # Manually pad or truncate to exactly 30 seconds
            if len(array) < self.max_samples:
                pad_width = self.max_samples - len(array)
                array = np.pad(array, (0, pad_width), mode="constant")
            else:
                array = array[:self.max_samples]

            padded_audios.append(array)
            texts.append(text)

        inputs = self.processor(
            padded_audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )

        labels = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=448 
        ).input_ids

        inputs["labels"] = labels
        return inputs
# Training arguments
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/model/whisper-azb-finetuned-20250717",
    per_device_train_batch_size=8,
    num_train_epochs=10,
    fp16=True,
    save_strategy="epoch",          # ✅ Save at the end of each epoch
    save_total_limit=1,             # ✅ Keep only the latest/best checkpoint
    evaluation_strategy="epoch",    # ✅ Required to evaluate and save best model
    load_best_model_at_end=True,    # ✅ Load the best checkpoint after training
    metric_for_best_model="loss",   # ✅ Optional: can also use "wer" or "cer" if defined
    greater_is_better=False,        # ✅ Lower loss is better
    logging_steps=10,
    learning_rate=1e-4,
    predict_with_generate=True,
    generation_max_length=225,
    remove_unused_columns=False,
)


# Trainer with dynamic collator
data_collator = WhisperDataCollator(processor)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # ✅ now works with evaluation_strategy
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
)

# Train and save
trainer.train()
model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)