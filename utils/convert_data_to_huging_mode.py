from datasets import Dataset, DatasetDict, Audio, concatenate_datasets
import os
from load_data import create_tts_dataset, convert_to_huggingface_dataset, features
import gc  # For garbage collection

# Paths and constants
MAIN_DIR = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset"
SPLIT_SIZE = 1000  # Process 1000 files at a time

def process_in_batches(directory, sampling_rate, split_size):
    """
    Process dataset in batches to avoid memory issues.
    Args:
        directory (str): Main directory containing audio files.
        sampling_rate (int): Sampling rate for audio files.
        split_size (int): Number of files to process in each batch.
    Returns:
        Dataset: Hugging Face Dataset object.
    """
    all_datasets = []
    dataset_list = create_tts_dataset(directory, sampling_rate=sampling_rate)

    # Process files in batches
    for i in range(0, len(dataset_list), split_size):
        print(f"Processing batch {i//split_size + 1}/{len(dataset_list)//split_size + 1}")
        batch = dataset_list[i:i+split_size]

        # Convert batch to Hugging Face Dataset
        hf_dataset = convert_to_huggingface_dataset(batch, features)
        all_datasets.append(hf_dataset)

        # Free up memory
        del batch
        gc.collect()

    # Combine all batches into one dataset using concatenate_datasets
    combined_dataset = concatenate_datasets(all_datasets)

    return combined_dataset

print("Step 1: Processing dataset in batches...")
tts_dataset = process_in_batches(MAIN_DIR, sampling_rate=16000, split_size=SPLIT_SIZE)

print("Step 2: Shuffling and splitting dataset...")
tts_dataset = tts_dataset.shuffle(seed=42)
train_test_split = tts_dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = train_test_split['test'].train_test_split(test_size=0.5, seed=42)

dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'validation': test_valid_split['train'],
    'test': test_valid_split['test']
})

print("Step 3: Saving processed dataset...")
dataset_dict.save_to_disk(os.path.join(MAIN_DIR, 'tts_dataset_hf'))

# Free memory explicitly
del tts_dataset, train_test_split, test_valid_split
gc.collect()
