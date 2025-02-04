import torch
import torchaudio
from transformers import WhisperFeatureExtractor, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration

# Step 1: Load Custom Tokenizer and Feature Extractor
marian_tokenizer = AutoTokenizer.from_pretrained("jalilkartal/marian-mt-en-azb")

# Extract Marian tokenizer vocabulary
marian_vocab = marian_tokenizer.get_vocab()

# Inspect the first few tokens
for i, (token, token_id) in enumerate(marian_vocab.items()):
    if i < 10:
        print(f"Token: {token}, ID: {token_id}")

from transformers import WhisperTokenizer

# Load Whisper tokenizer
whisper_tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")

# Replace Whisper vocabulary with Marian vocabulary
whisper_tokenizer.vocab = marian_vocab
whisper_tokenizer.ids_to_tokens = {v: k for k, v in marian_vocab.items()}
whisper_tokenizer.tokens_to_ids = marian_vocab

# Save the updated tokenizer for future use
whisper_tokenizer.save_pretrained("whisper-small-en-azb-tokenizer")
print("Updated Whisper tokenizer saved successfully.")

from transformers import WhisperProcessor, WhisperFeatureExtractor

# Load the Whisper feature extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# Load the updated tokenizer
updated_tokenizer = WhisperTokenizer.from_pretrained("whisper-small-en-azb-tokenizer")

# Create a new WhisperProcessor
processor = WhisperProcessor(tokenizer=updated_tokenizer, feature_extractor=feature_extractor)
processor.save_pretrained("whisper-small-en-azb-tokenizer")


"""
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

# Step 2: Create a Processor with the Custom Tokenizer
processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)

# Step 3: Load the Whisper Model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")

# Step 4: Load and Preprocess Audio File
audio_path = "1.mp3"  # Path to your audio file
waveform, sample_rate = torchaudio.load(audio_path)

# Resample audio to 16kHz if needed
resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Ensure audio has the expected shape: (batch_size, num_channels, num_samples)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0).unsqueeze(0)

# Preprocess the audio into features
inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

# Step 5: Generate Transcription
with torch.no_grad():
    predicted_ids = model.generate(inputs["input_features"])

# Step 6: Decode Transcription
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("Transcription:", transcription)
"""