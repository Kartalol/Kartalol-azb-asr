# pip install pydub

import os
from pydub import AudioSegment


def process_audio_files(root_dir, sampling_rate=16000, channels=1):
    """
    Traverse folders, process WAV files, and overwrite them with optimized versions.
    
    Args:
        root_dir (str): Root directory containing WAV files.
        sampling_rate (int): Desired sampling rate (e.g., 16000 Hz).
        channels (int): Desired number of audio channels (1 for mono).
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):  # Process only .wav files
                file_path = os.path.join(subdir, file)
                try:
                    print(f"Processing: {file_path}")
                    
                    # Load the audio file
                    audio = AudioSegment.from_wav(file_path)
                    
                    # Adjust sampling rate and channels
                    processed_audio = audio.set_frame_rate(sampling_rate).set_channels(channels)
                    
                    # Export and overwrite the original file
                    processed_audio.export(file_path, format="wav")
                    print(f"Saved: {file_path}")
                
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Specify the root directory containing your audio files
root_directory = "/path/to/your/audio/folders"

# Process all WAV files in the directory
process_audio_files(root_directory)
