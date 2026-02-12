import json
import librosa
import torch
import numpy as np
import optuna
from transformers import AutoProcessor, AutoModelForCTC
from jiwer import wer
from tqdm import tqdm
import pickle  # For caching logits
import os

# Load the model & processor
processor = AutoProcessor.from_pretrained("./withlm")
model = AutoModelForCTC.from_pretrained("./withlm").to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load dataset (JSON file)
with open("data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

logits_cache_file = "logits_cache.pkl"

# ================================
# 🔹 STEP 1: Compute & Cache Logits
# ================================
if not torch.cuda.is_available():
    print("⚠️ CUDA not available, running on CPU (this will be slower).")


def compute_and_cache_logits():
    """Run inference one-by-one and store logits."""
    logits_cache = {}

    for sample in tqdm(dataset, desc="Computing Logits"):
        audio_path = os.path.join(
            "/home/davud/Desktop/storage/webspeech/azeri-test/audio", sample["audio"]
        )
        audio, sr = librosa.load(audio_path, sr=16000)

        input_values = processor(audio, sampling_rate=sr, return_tensors="pt").to(
            model.device
        )

        with torch.no_grad():
            logits = (
                model(**input_values).logits.cpu().numpy()[0]
            )  # Move to CPU immediately

        logits_cache[audio_path] = logits  # Store on CPU, not GPU

        torch.cuda.empty_cache()  # Free GPU memory after each iteration

    with open(logits_cache_file, "wb") as f:
        pickle.dump(logits_cache, f)  # Save to disk

    print("✅ Logits cached successfully!")


# Compute & cache logits if not already saved
try:
    with open(logits_cache_file, "rb") as f:
        logits_cache = pickle.load(f)
    print("🔄 Loaded cached logits.")
except FileNotFoundError:
    print("⏳ Logits cache not found. Running inference...")
    compute_and_cache_logits()
    with open(logits_cache_file, "rb") as f:
        logits_cache = pickle.load(f)

# ================================
# 🔹 STEP 2: Optimize Alpha & Beta
# ================================


def objective(trial):
    """Optuna objective function to optimize alpha & beta."""
    alpha = trial.suggest_float("alpha", 0.1, 3.0)
    beta = trial.suggest_float("beta", 0.1, 5.0)

    total_wer = 0
    count = 0

    for sample in dataset:
        # audio_path = sample["audio"]
        audio_path = os.path.join(
            "/home/davud/Desktop/storage/webspeech/azeri-test/audio", sample["audio"]
        )
        ground_truth = sample["text"]

        logits = logits_cache[audio_path]  # Load cached logits
        predicted_text = processor.decode(logits, alpha=alpha, beta=beta).text

        error = wer(ground_truth, predicted_text)
        total_wer += error
        count += 1

    return total_wer / count  # Minimize WER


# Run Optuna optimization
study = optuna.create_study(direction="minimize")  # Minimize WER
study.optimize(objective, n_trials=50)  # Run 50 trials

# Best hyperparameters
best_alpha = study.best_params["alpha"]
best_beta = study.best_params["beta"]
print(
    f"🎯 Best Alpha: {best_alpha}, Best Beta: {best_beta}, Best WER: {study.best_value}"
)

# Save best parameters
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)
