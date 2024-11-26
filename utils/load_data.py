# 1- read voices: write a programe to read all voices from all folders of dataset
# 2- read texts or books
# 3- match voices with text

from datasets import Features, Value, Array3D, Sequence
import os
import pandas as pd
import librosa
import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, Sequence
import string
import re

def get_person_folders(main_dir):
    """Retrieve all person folders ending with '-bot'."""
    return [
        os.path.join(main_dir, folder)
        for folder in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, folder)) and folder.endswith('-bot')
    ]


def normalize_text(text):
    """Normalize text by lowercasing and removing punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())  # Remove extra whitespace
    return text


def load_csv_files(name_csv_dir):
    """Load CSV files into a dictionary with PersonID as keys."""
    csv_dict = {}
    for file in os.listdir(name_csv_dir):
        if file.endswith('.csv'):
            person_id = os.path.splitext(file)[0]
            # print(person_id)
            csv_path = os.path.join(name_csv_dir, file)
            try:
                df = pd.read_csv(csv_path)
                csv_dict[person_id] = df
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
    return csv_dict

def create_tts_dataset(main_dir, sampling_rate=22050):
        dataset = []
        person_folders = get_person_folders(os.path.join(main_dir, 'voices'))
        name_csv_dir = os.path.join(main_dir, 'sentences')
        csv_dict = load_csv_files(name_csv_dir)

        for person_folder in person_folders:
            # Extract PersonID from folder name (assuming format 'PersonID-bot')
            folder_name = os.path.basename(person_folder)
            print(folder_name, " : folder_name")
            person_id = folder_name.replace('-bot', '')

            if person_id not in csv_dict:
                print(f"CSV for PersonID {person_id} not found. Skipping this person.")
                continue

            df = csv_dict[person_id]

            # Iterate over all wav files in the person folder
            for file in os.listdir(person_folder):
                if file.endswith('.wav'):
                    # Expected filename format: "PersonID_SentenceID_Number_BookID.wav"
                    try:
                        parts = os.path.splitext(file)[0].split('_')
                        if len(parts) != 4:
                            print(f"Filename {file} does not match the expected format. Skipping.")
                            continue
                        file_person_id, sentence_id, number, book_id = parts
                        match = re.search(r'ID(\d+)', sentence_id)
                        if match:
                            sentence_id = int(match.group(1))
                        if 'f' in file_person_id:
                          gender = 'female'
                        else:
                          gender = 'male'

                    except Exception as e:
                        print(f"Error parsing filename {file}: {e}")
                        continue

                    # Verify PersonID matches
                    if file_person_id != person_id:
                        print(f"PersonID mismatch in file {file}. Skipping.")
                        continue

                    # Find the corresponding text in the CSV
                    # Assuming that SentenceID, Number, and BookID uniquely identify a row
                    matching_rows = df.iloc[sentence_id]
                    if matching_rows.empty:
                        print(f"No matching text found for file {file}. Skipping.")
                        continue

                    # text = matching_rows.iloc[0]['Sentence']
                    text = matching_rows['Sentence']
                    normalized = normalize_text(text)

                    # Define the path to the audio file
                    audio_path = os.path.join(person_folder, file)

                    # Load audio file
                    try:
                        audio_array, sr = librosa.load(audio_path, sr=sampling_rate)
                        # Optionally, you can apply padding or trimming here
                    except Exception as e:
                        print(f"Error loading audio file {audio_path}: {e}")
                        continue

                    # Append to dataset
                    dataset.append({
                        'person_id': person_id,
                        'gender': gender,
                        'sentence_id': sentence_id,
                        'number_in_book': number,
                        'book_id': book_id,
                        'file': file,
                        'path': audio_path,
                        'audio2': audio_array,  # Keep as NumPy array
                        'audio': audio_path,  # Store path instead of array
                        'sampling_rate': sr,
                        'sentence': text,
                        'normalized_text': normalized
                    })

        return dataset


# Define the features
features = Features({
    'person_id': Value('string'),
    'gender': Value('string'),
    'sentence_id': Value('int32'),
    'number_in_book': Value('string'),
    'book_id': Value('string'),
    'file': Value('string'),
    'path': Value('string'),
    'audio2': Sequence(Value('float32')),  # Keep as NumPy array
    'audio': Value('string'),
    'sampling_rate': Value('int32'),
    'sentence': Value('string'),
    'normalized_text': Value('string'),
})

def convert_to_huggingface_dataset(dataset_list, features):
    # Convert the list of dicts into a Hugging Face Dataset
    dataset = Dataset.from_list(dataset_list, features=features)
    return dataset


if __name__ == "__main__":
    # Step 1: Create the dataset list. Select sample rate here/
    dataset_list = create_tts_dataset(MAIN_DIR, sampling_rate=22050)

    # Step 2: Define features (already done above)

    # Step 3: Convert to Hugging Face Dataset
    tts_dataset = convert_to_huggingface_dataset(dataset_list, features)

    # (Optional) Shuffle the dataset
    tts_dataset = tts_dataset.shuffle(seed=42)

    # Step 4: (Optional) Split into train, validation, test
    # For example, 80% train, 10% validation, 10% test
    train_testvalid = tts_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['test'],
        'test': test_valid['train']
    })

    # Step 5: Save the dataset to disk (optional)
    dataset_dict.save_to_disk(os.path.join(MAIN_DIR, 'tts_dataset_hf'))
    print("Dataset successfully saved to disk.")

    # Step 6: Loading the dataset later
    # loaded_dataset = DatasetDict.load_from_disk(os.path.join(MAIN_DIR, 'tts_dataset_hf'))

    # Example: Accessing the first instance
    dataset_dict['test'][0]

