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


def check_data_reading(dataset):
    """Validate dataset by printing a few samples."""
    for idx in range(min(5, len(dataset))):  # Check up to 5 samples
        sample = dataset[idx]
        audio_path = dataset.audio_files[idx]
        file_id = os.path.splitext(os.path.basename(audio_path))[0].split("-")[0]
        sentence = dataset.id_to_sentence[file_id]

        print(f"Sample {idx + 1}:")
        print(f"  File ID: {file_id}")
        print(f"  Audio Path: {audio_path}")
        print(f"  Corresponding Sentence: {sentence}")
        print(f"  Input Features Shape: {sample['input_features'].shape}")
        print(f"  Labels Shape: {sample['labels'].shape}")
        print("-" * 50)


if __name__ == "__main__":
    # Directories and CSV path
    audio_dir = "../../dataset/Kartal Ol Gold Test set/voices"
    excel_path = "../../dataset/Kartal Ol Gold Test set/sentences.xlsx"
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

    # Create dataset and DataLoader
    dataset = AudioTextDataset(audio_dir, excel_path, processor)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Validate Dataset Reading
    print(f"Loaded dataset with {len(dataset)} samples.\n")
    check_data_reading(dataset)
