# © 2026 Kartal Ol NGO. Confidential and proprietary.
# Internal use only. Unauthorized use or distribution is prohibited.

from jiwer import wer, cer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import pandas as pd
import os
import json

def evaluation(base_path, model_dir):
    output_path = os.path.join(base_path, "kartalol_gold_testset/results.csv")
    grand_truth_path = os.path.join(base_path, "kartalol_gold_testset/meta_data.json")
    with open(grand_truth_path, "r", encoding="utf-8") as f:
        json_info = json.load(f)
    results = []
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    for i, audio_file_name in enumerate(json_info["data"]):
        print(i)
        try:
            audio, _ = librosa.load(
                os.path.join(base_path, "kartalol_gold_testset/voices", audio_file_name['audio_filepath']), 
                sr=16000)
        except:
            result = {
                "voice_id":audio_file_name['audio_filepath'],
                "duration": audio_file_name['duration'],
                "grand_truth": audio_file_name['text'],
                "prediction": None,
                }
            results.append(result)
            continue
        # Specify the language
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            task="transcribe",  # Not "translate"
            return_attention_mask=False,
        )
        #breakpoint()
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features.to(device))
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            result = {
                "voice_id":audio_file_name['audio_filepath'],
                "duration": audio_file_name['duration'],
                "grand_truth": audio_file_name['text'],
                "prediction": transcription,
                "wer": wer(audio_file_name['text'], transcription),
                "cer": cer(audio_file_name['text'], transcription)
                }
        results.append(result)    
    pd.DataFrame(results).to_csv(output_path)

def results_analysis(base_path):
    output_path = os.path.join(base_path, "kartalol_gold_testset/results.csv")
    df = pd.read_csv(output_path)
    error = wer(df['grand_truth'].tolist(), df['prediction'].tolist())
    print(f"WER: {error:.2%}")
    cer_error = cer(df['grand_truth'].tolist(), df['prediction'].tolist())
    print(f"CER: {cer_error:.2%}")

if __name__ == "__main__":
    # Paths
    base_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset"
    model_dir = os.path.join(base_path, "model/whisper-azb-finetuned-20250717") # directory where your model is saved
    # path to your test audio file
    evaluation(base_path, model_dir)
    results_analysis(base_path)