import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def test_audio_to_text(audio_path, processor_path, model_path, sampling_rate=16000):
    """
    Converts audio to text using a fine-tuned Whisper model.
    
    Args:
        audio_path: Path to the audio file to transcribe.
        processor_path: Path to the saved WhisperProcessor.
        model_path: Path to the fine-tuned Whisper model.
        sampling_rate: Sampling rate for audio processing (default: 16kHz).
    
    Returns:
        Transcribed text from the audio file.
    """
    # Load the fine-tuned processor and model
    processor = WhisperProcessor.from_pretrained(processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Ensure mono audio
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)

    # Resample to the target sampling rate
    if sample_rate != sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sampling_rate)
        waveform = resampler(waveform)

    # Normalize waveform to range [-1, 1]
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = waveform.squeeze(0).numpy()

    # Extract input features
    inputs = processor.feature_extractor(
        raw_speech=waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )
    input_features = inputs.input_features

    # Explicitly create attention mask for input features
    attention_mask = torch.ones(input_features.shape, dtype=torch.long)

    # Generate transcription
    model.eval()
    with torch.no_grad():
        predicted_ids = model.generate(input_features, attention_mask=attention_mask)

    # Decode transcription
    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


if __name__ == "__main__":
    # Paths to your fine-tuned model and processor
    processor_path = "./whisper-azb-finetuned"
    model_path = "./whisper-azb-finetuned"
    
    # Path to the audio file to test
    audio_path = "3.ogg"
    
    # Transcribe the audio file
    transcription = test_audio_to_text(audio_path, processor_path, model_path)
    
    # Print the transcription
    print("Transcription:", transcription)
