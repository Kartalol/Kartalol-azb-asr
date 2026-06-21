from datasets import load_from_disk, Audio
import os
import re
import json
import numpy as np
from transformers import (
    Wav2Vec2CTCTokenizer, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC, 
    TrainingArguments, 
    Trainer
)
from dataclasses import dataclass
from typing import List, Dict, Union, Optional
import torch
import evaluate
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("# Load the preprocessed dataset from disk")
MAIN_DIR = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset"
dataset_path = os.path.join(MAIN_DIR, 'tts_dataset_hf')
dataset_dict = load_from_disk(dataset_path)

# Split dataset into train, validation, and test
train_dataset = dataset_dict['train']
valid_dataset = dataset_dict['validation']
test_dataset = dataset_dict['test']

# Show random samples (optional)
def show_random_elements(dataset, num_examples=10):
    import random
    import pandas as pd
    picks = random.sample(range(len(dataset)), num_examples)
    df = pd.DataFrame(dataset.select(picks))
    print(df)

# Preprocess sentences by removing special characters
chars_to_ignore_regex = '[،؟!.؛:\"“%‘”�]'
def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    return batch

train_dataset = train_dataset.map(remove_special_characters)
valid_dataset = valid_dataset.map(remove_special_characters)
test_dataset = test_dataset.map(remove_special_characters)
# Extract vocabulary
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = train_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train_dataset.column_names)
vocab_test = test_dataset.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test_dataset.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

print("# Save vocabulary to a JSON file")
vocab_path = os.path.join(MAIN_DIR, "vocab.json")
with open(vocab_path, 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

print("# Initialize tokenizer and processor")
tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# Cast audio column for efficient processing
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
valid_dataset = valid_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Prepare datasets for training
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4)
valid_dataset = valid_dataset.map(prepare_dataset, remove_columns=valid_dataset.column_names, num_proc=4)
test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, num_proc=4)

# Define a data collator for dynamic padding
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# Define evaluation metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}
# Load model directly
print("# Load pre-trained Wav2Vec2 model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
).to(device)
model.gradient_checkpointing_enable()

model.freeze_feature_extractor()
print("# Training arguments")
training_args = TrainingArguments(
    output_dir=os.path.join(MAIN_DIR, "wav2vec2-finetuned-azb"),
    group_by_length=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    evaluation_strategy="epoch",
    num_train_epochs=200,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=10,
    learning_rate=3e-4,
    warmup_steps=500,
    save_total_limit=2,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor,
)

print("# Train the model")
trainer.train()

# Save the final model
model.save_pretrained(os.path.join(MAIN_DIR, "wav2vec2-finetuned-azb"))
processor.save_pretrained(os.path.join(MAIN_DIR, "wav2vec2-finetuned-azb"))
