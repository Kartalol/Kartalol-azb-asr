# © 2026 Kartal Ol NGO. Confidential and proprietary.
# Internal use only. Unauthorized use or distribution is prohibited.


from jiwer import wer, cer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa
import psutil
import os
import time

def get_resource_usage():
    """Returns the current CPU and RAM usage of the process."""
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=None) # Non-blocking
    ram_info = process.memory_info()
    return cpu_percent, ram_info.rss / (1024 * 1024) # RSS in MB

# Paths
model_dir = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/model/whisper-azb-finetuned-base-20251104"
audio_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/kartalol_gold_testset/voices/278-m-jalil-audio_2024-05-25_22-53-21.ogg"

# Initial resource usage
initial_cpu, initial_ram = get_resource_usage()
print(f"Initial State: CPU: {initial_cpu:.2f}%, RAM: {initial_ram:.2f} MB")

# Load model and processor
processor = WhisperProcessor.from_pretrained(model_dir)
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
model.eval()

# Resource usage after loading the model
model_load_cpu, model_load_ram = get_resource_usage()
print(f"After Model Load: CPU: {model_load_cpu:.2f}%, RAM: {model_load_ram:.2f} MB")
print(f"    (Model Load delta: CPU: {model_load_cpu - initial_cpu:.2f}%, RAM: {model_load_ram - initial_ram:.2f} MB)")

# Load audio
audio, sr = librosa.load(audio_path, sr=16000)

# Specify the language
inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    task="transcribe",
)

device = "cpu" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("\n--- Starting Transcription ---")
start_time = time.time()
# Resource usage just before generation
pre_gen_cpu, pre_gen_ram = get_resource_usage()
print(f"Pre-Generation: CPU: {pre_gen_cpu:.2f}%, RAM: {pre_gen_ram:.2f} MB")

with torch.no_grad():
    # The generation process is where the heavy lifting occurs.
    predicted_ids = model.generate(inputs.input_features.to(device))
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Final resource usage and elapsed time
end_time = time.time()
final_cpu, final_ram = get_resource_usage()
elapsed_time = end_time - start_time

print("--- Transcription Complete ---")
print(f"Transcription Time: {elapsed_time:.2f} seconds")
print(f"Final State: CPU: {final_cpu:.2f}%, RAM: {final_ram:.2f} MB")
print(f"    (Generation delta: CPU: {final_cpu - pre_gen_cpu:.2f}%, RAM: {final_ram - pre_gen_ram:.2f} MB)")

print("\nTranscription:", transcription)