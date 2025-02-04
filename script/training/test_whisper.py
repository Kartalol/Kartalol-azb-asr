import torchaudio
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

def test_audio_to_text(audio_path, model_id, device="cuda", sampling_rate=16000, language="az"):
    """
    Converts audio to text using a fine-tuned Whisper model with pipeline.

    Args:
        audio_path: Path to the audio file to transcribe.
        model_id: Path or ID of the fine-tuned model.
        device: Device to use for inference ("cuda" or "cpu").
        sampling_rate: Sampling rate for audio processing (default: 16kHz).
        language: Language for transcription (default: Azerbaijani "az").

    Returns:
        Transcribed text from the audio file.
    """
    # Load the model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Set the language for transcription using the model's config
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids

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

    # Use pipeline for transcription
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device == "cuda" else -1,
    )

    # Perform transcription
    result = pipe(waveform.squeeze(0).numpy())
    return result["text"]


if __name__ == "__main__":
    # Path to your fine-tuned model and processor
    model_id = "./whisper-azb-finetuned-mask-attention"
    
    # Path to the audio file to test
    audio_path = "3.ogg"
    
    # Transcribe the audio file in Azerbaijani
    transcription = test_audio_to_text(audio_path, model_id, language="az")
    
    # Print the transcription
    print("Transcription:", transcription)
