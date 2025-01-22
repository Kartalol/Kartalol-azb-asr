import os
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

import os
import torchaudio
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor

class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, excel_path, processor, sampling_rate=16000):
        """
        Args:
            audio_dir: Directory containing audio files.
            excel_path: Path to the Excel file with 'id' and 'sentence' columns.
            processor: WhisperProcessor for preprocessing.
            sampling_rate: Target sampling rate for audio files.
        """
        self.audio_dir = audio_dir
        self.processor = processor
        self.sampling_rate = sampling_rate

        # Load Excel file
        self.data = pd.read_excel(excel_path)

        # Ensure required columns exist
        if "id" not in self.data.columns or "azb" not in self.data.columns:
            raise ValueError("The Excel file must contain 'id' and 'azb' columns.")

        # Create a mapping of id to sentence
        self.id_to_sentence = dict(zip(self.data["id"].astype(str), self.data["azb"]))

        # Get all audio files with .ogg format
        self.audio_files = [
            os.path.join(audio_dir, f)
            for f in os.listdir(audio_dir)
            if f.endswith(".ogg")
        ]

        # Ensure that ids in filenames match the Excel file
        self.audio_files = [
            f for f in self.audio_files if os.path.splitext(os.path.basename(f))[0].split("-")[0] in self.id_to_sentence
        ]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Get audio file path
        audio_path = self.audio_files[idx]

        # Extract id from filename
        file_id = os.path.splitext(os.path.basename(audio_path))[0].split("-")[0]

        # Get corresponding sentence
        sentence = self.id_to_sentence[file_id]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        # Resample to 16kHz if necessary
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # Normalize waveform to range [-1, 1]
        waveform = waveform / torch.max(torch.abs(waveform))
        waveform = waveform.squeeze(0).numpy()

        # Extract input features
        input_features = self.processor.feature_extractor(
            raw_speech=waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features

        # Tokenize the transcription
        labels = self.processor.tokenizer(
            text=sentence,
            return_tensors="pt",
            padding=True
        ).input_ids

        return {
            "input_features": input_features.squeeze(0),  # log-Mel spectrogram
            "labels": labels.squeeze(0)  # Tokenized transcription
        }

# Load the Whisper feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

# Load the updated tokenizer
updated_tokenizer = WhisperTokenizer.from_pretrained("/content/drive/MyDrive/KartalOl Corpus/Models/whisper-en-azb-tokenizer")

# Create a new WhisperProcessor
processor = WhisperProcessor(tokenizer=updated_tokenizer, feature_extractor=feature_extractor)

# Load Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

# Resize token embeddings to match the new tokenizer
model.resize_token_embeddings(len(updated_tokenizer))
print("Updated Whisper model initialized successfully.")


# ---------------------
# Step 2: Create Dataset and DataLoader
# ---------------------
# Directories and CSV path
audio_dir = "/content/drive/MyDrive/KartalOl Corpus/Dataset/Speech Recognition dataset/voices"
excel_path = "/content/drive/MyDrive/KartalOl Corpus/Dataset/Speech Recognition dataset/sentences.xlsx"
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

# Create dataset and DataLoader
dataset = AudioTextDataset(audio_dir, excel_path, processor)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ---------------------
# Step 3: Training Arguments
# ---------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-azb-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="no",  # Disable evaluation
    num_train_epochs=5,
    save_steps=500,
    learning_rate=5e-5,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",
    logging_steps=50
)
# ---------------------
# Step 4: Trainer Setup
# ---------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,  # Tokenizer for decoding
    data_collator=lambda data: {
        "input_features": torch.stack([f["input_features"] for f in data]),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in data], batch_first=True, padding_value=-100
        )
    },
)

# ---------------------
# Step 5: Fine-Tuning
# ---------------------
trainer.train()

# Save the model
processor.save_pretrained("./whisper-azb-finetuned")
print("Fine-tuning completed and model saved!")