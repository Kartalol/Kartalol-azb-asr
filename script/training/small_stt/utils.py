import pandas as pd
from datasets import load_dataset
import json 
import os
from pydub import AudioSegment

def convert_tsv_to_csv(tsv_filename="train.tsv", columns_name=["sentence"]):
    """
    Convert a TSV file to CSV format.
    Assumes the TSV file is named 'train.tsv' and contains a column named 'sentence'.
    The output will be saved as 'sentences.csv'.
    """
    # Read the TSV file
    df = pd.read_csv(tsv_filename, sep="\t")
    #breakpoint()

    # Select the column you want, e.g., "sentence"
    selected = df[columns_name]

    return selected


def concatinate_all_dataset():
    pass

def read_excel_into_json(excel_path, json_path, output_path):
    # Step 1: Load Excel and convert to JSON-style list
    df = pd.read_excel(excel_path)

    new_data = []
    for _, row in df.iterrows():
        new_data.append({
            "audio_filepath": row["audio_filepath"],
            "duration": row["duration"],
            "text": row["azb"]  # Only keeping azb text
        })

    # Step 2: Load the old JSON
    with open(json_path, "r", encoding="utf-8") as f:
        old_json = json.load(f)

    # Step 2.5: Filter old data based on sentence length
    filtered_old_data = [
        item for item in old_json["data"]
        if len(item["text"].split()) > 7
    ]

    # Step 3: Combine filtered old data + new data
    combined_data = {
        "data": filtered_old_data + new_data
    }

    # Step 4: Save the combined JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)


def read_csv_into_json(csv_path, json_path, output_path):
    # Step 1: Load Excel and convert to JSON-style list
    df = pd.read_csv(csv_path)

    new_data = []
    for _, row in df.iterrows():
        if len(row["sentence"].split(" ")) > 7:
            new_data.append({
                "audio_filepath": row["path"],
                "duration": 0,
                "text": row["sentence"]  # Only keeping azb text
            })

    # Step 2: Load the old JSON
    with open(json_path, "r", encoding="utf-8") as f:
        old_json = json.load(f)

    # Step 3: Combine old + new data
    combined_data = {
        "data": old_json["data"] + new_data
    }

    # Step 4: Save the combined JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)


def json_analysis(json_path):
    # Step 2: Load the old JSON
    with open(json_path, "r", encoding="utf-8") as f:
        json_info = json.load(f)
    print(len(json_info['data']))
    total_duration = sum(item["duration"] for item in json_info["data"])
    print(f"Total duration: {total_duration:.2f} seconds")
 
def check_missing_files(json_path,audio_dir):
    # Load your JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)["data"]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Build full path and check file existence
    df["full_path"] = df["audio_filepath"].apply(lambda x: os.path.join(audio_dir, x))
    df["exists"] = df["full_path"].apply(os.path.exists)

    # Separate missing and existing files
    missing = df[~df["exists"]]
    print(missing)
    breakpoint()

import os
import json
import glob
from pydub.utils import mediainfo
from tqdm import tqdm
def concatinate_json_files(json_folder, audio_folder, output_path, target_file):

    # Collect all .json files
    json_files = glob.glob(os.path.join(json_folder, "*.json"))

    merged_data = []

    for file_path in tqdm(json_files, desc="Processing JSON files"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            for entry in content.get("data", []):
                trans = entry.get("transcription", "").strip()
                clip_id = str(entry.get("id", "")).strip()
                mp3_filename = f"{clip_id}.mp3"
                mp3_path = os.path.join(audio_folder, mp3_filename)

                # Try to get duration
                try:
                    info = mediainfo(mp3_path)
                    duration = float(info["duration"])
                except Exception as e:
                    print(f"⚠️ Failed to get duration for {mp3_filename}: {e}")
                    duration = None

                merged_data.append({
                    "audio_filepath": mp3_filename,
                    "duration": duration,
                    "text": trans
                })
    df = pd.DataFrame(merged_data)

    # Save to CSV
    df.to_csv(target_file, index=False, encoding="utf-8")
    """
    # Step 2: Load the old JSON
    with open(output_path, "r", encoding="utf-8") as f:
        old_json = json.load(f)

    # Step 3: Combine old + new data
    combined_data = {
        "data": old_json["data"] + merged_data
    }

    # Step 4: Save the combined JSON
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    """
    print(f"✅ Done! Merged entries into 'merged_transcriptions.json'")

if __name__ == "__main__":
    audio_dir = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/audio/output_audio/"

    # json_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/training_20250702.json"
    # check_missing_files(json_path, audio_dir)
    #json_analysis(json_path)
    # How to concatinate json and xlsx
    # excel_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/VoxLingua107_AZ_AZB.xlsx"
    # csv_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/common_voices_az_azb_sentences.csv"
    # json_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/training_20250702.json"
    output_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/training_20250703.json"
    json_folder = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/json/json_files/"
    target_file = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/sentences/merged_transcriptions.csv"
    # #read_excel_into_json(excel_path,json_path, output_path)
    # read_csv_into_json(csv_path, json_path, output_path)
    concatinate_json_files(json_folder, audio_dir, output_path, target_file)
    pass