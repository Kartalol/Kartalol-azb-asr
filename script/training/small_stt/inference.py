# © 2026 Kartal Ol NGO. Confidential and proprietary.
# Internal use only. Unauthorized use or distribution is prohibited.


from jiwer import wer, cer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import pandas as pd
# Paths
model_dir = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/model/whisper-azb-finetuned-20250717" # directory where your model is saved
audio_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/kartalol_gold_testset/voices/278-m-jalil-audio_2024-05-25_22-53-21.ogg"
# path to your test audio file

processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
model.eval()

audio, sr = librosa.load(audio_path, sr=16000)
# Specify the language
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    task="transcribe",  # Not "translate"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device) 

with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features.to(device))
    #breakpoint()
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
gt = pd.read_csv("../../../dataset/sentences/common_voices_az_azb_sentences.csv")
autdio_name = audio_path.split("/")[-1]
print("Transcription:", transcription)
ground_truth = gt[gt["path"] == autdio_name]["sentence"].values[0]

error = wer(ground_truth, transcription)
print(f"WER: {error:.2%}")
cer_error = cer(ground_truth, transcription)
print(f"CER: {cer_error:.2%}")
#Transcription: اینسان جمعیتی‌نین فعّالیّتی مین ایللر عرضینده طبیعی صۇرتده ایقلیمین دییشمه‌سینه گتیریب چؽخارمیشدیر.
