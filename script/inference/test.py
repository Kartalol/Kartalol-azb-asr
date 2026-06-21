import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

# Path to the saved model and processor
MAIN_DIR = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset"

model_dir = os.path.join(MAIN_DIR, "wav2vec2-finetuned-azb")

# Load fine-tuned model and processor
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")

def transcribe_audio(audio_path, processor, model, sampling_rate=16000):
    """
    Transcribes a given .wav audio file to text.
    
    Args:
        audio_path (str): Path to the audio file.
        processor (Wav2Vec2Processor): Pretrained processor.
        model (Wav2Vec2ForCTC): Fine-tuned Wav2Vec2 model.
        sampling_rate (int): Sampling rate for audio processing.
    
    Returns:
        str: Transcribed text from the audio.
    """
    # Load the audio file
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample to match the model's expected sampling rate
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        waveform = resampler(waveform)

    # Process the audio to input format
    input_values = processor(waveform.squeeze().numpy(), sampling_rate=sampling_rate, return_tensors="pt").input_values

    # Move to device (GPU/CPU)
    input_values = input_values.to("cuda" if torch.cuda.is_available() else "cpu")

    # Perform inference
    model.eval()
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode predicted tokens to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# Test the transcription function
audio_file = "4.wav"  # Replace with your .wav file path
transcription = transcribe_audio(audio_file, processor, model)
print(f"Transcription: {transcription}")
