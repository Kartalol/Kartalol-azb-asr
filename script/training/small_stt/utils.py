import pandas as pd
import json 
import os
import glob
from pydub.utils import mediainfo
from tqdm import tqdm
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


def concatinate_json_files(json_folder, audio_folder, output_path, target_file):
    """
    Read all json files and save it in csv files, 
    it is poosible make unite json file as well.
    """
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

def check_random_sentences(csv_path):

    df = pd.read_csv(csv_path)
    for i, line in enumerate(df.iterrows()):
        if i % 800 == 0:
            print(f"{line[1]['text']}\n")
            print(f"{line[1]['azb']}\n")
        if i > 60000:
            break

def convert_xlsx_json(xlsx_path, audio_dir,output_json_path):

    # === READ EXCEL ===
    df = pd.read_excel(xlsx_path)
    df = df.dropna(subset=["id", "azb"])  # drop rows with missing ID or text
    df["id"] = df["id"].astype(str)       # ensure IDs are strings for comparison

    # === BUILD OUTPUT ===
    data = []
    supported_exts = (".wav", ".mp4", ".ogg",".mp3")
    
    for fname in os.listdir(audio_dir):
        
        if not fname.lower().endswith(supported_exts):
            continue
        
        file_id = fname.split("-")[0]  # assumes format like: "1_xyz.wav"
        match = df[df["id"] == file_id]
        if match.empty:
            continue

        text = match.iloc[0]["azb"]
        audio_path = os.path.join(audio_dir, fname)

        try:
            audio = AudioSegment.from_file(audio_path)
            duration = round(len(audio) / 1000.0, 4)
        except Exception as e:
            print(f"Skipping {audio_path} due to error: {e}")
            continue

        data.append({
            "audio_filepath": fname,
            "duration": duration,
            "text": text
        })

    # === SAVE JSON ===
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(data)} entries to {output_json_path}")

def df_nan_row(df_path):
    df = pd.read_csv(df_path)
    dff = df[df['prediction'].isna() | (df['prediction'].astype(str).str.strip() == "")]
    breakpoint()
    print(dff)
if __name__ == "__main__":
    base_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/"

    #audio_dir = os.path.join(base_path,"audio/output_audio/")
    # json_path = os.path.join(base_path,"sentences/training_20250702.json")
    # check_missing_files(json_path, audio_dir)
    # json_analysis(json_path)

    # How to concatinate json and xlsx
    # excel_path = os.path.join(base_path, "sentences/VoxLingua107_AZ_AZB.xlsx")
    # csv_path = os.path.join(base_path, "sentences/common_voices_az_azb_sentences.csv")
    # json_path = os.path.join(base_path, "sentences/training_20250702.json")
    # output_path = os.path.join(base_path, "sentences/training_20250703.json")
    # json_folder = os.path.join(base_path,"json/json_files/")
    #target_file = os.path.join(base_path, "sentences/merged_transcriptions.csv")
    # read_excel_into_json(excel_path,json_path, output_path)

    # read_csv_into_json(csv_path, json_path, output_path)

    #concatinate_json_files(json_folder, audio_dir, output_path, target_file)

    # csv_path = os.path.join(base_path, "sentences/merged_transcriptions.csv")
    # check_random_sentences(csv_path)

    # How to read from xlsx file get info and save in json file
    # xlsx_path = os.path.join(base_path, "kartalol_gold_testset/sentences.xlsx")
    # audio_dir = os.path.join(base_path, "kartalol_gold_testset/voices")
    # output_json_path = os.path.join(base_path, "kartalol_gold_testset/meta_data.json")
    # convert_xlsx_json(xlsx_path, audio_dir,output_json_path)

    #How to find nan row
    df_path = "/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/kartalol_gold_testset/results.csv"
    df_nan_row(df_path)