import numpy as np
from typing import Final
from telegram import Update, Document
from telegram.ext import MessageHandler, Filters, CallbackContext, Updater
from pydub import AudioSegment
import torch
from silero_vad import get_speech_timestamps

TOKEN: Final = 'TELEGRAM_BOT_TOKEN'
BOT_USERNAME: Final = 'TELEGRAM_BOT_ID'
pending_voice_messages = {}


# Function to convert OGG to MP3
# just in case of the requirement of format converting, not used right now
def convert_ogg_to_mp3(ogg_file_path: str, mp3_file_path: str):
    try:
        audio = AudioSegment.from_ogg(ogg_file_path)
        audio.export(mp3_file_path, format="mp3")
        print(f"Converted {ogg_file_path} to {mp3_file_path}")
    except Exception as e:
        print(f"Error converting file: {e}")

# handle voice messages, if the caption is available, saves with caption
# otherwise stores the voice and wait for a reply message on the voice
def handle_voice(update: Update, context: CallbackContext):
    msg = update.effective_message
    chat_id = msg.chat.id
    message_id = msg.message_id
    chat_title = msg.chat.title
    caption = update.message.caption

    if not caption:
        pending_voice_messages[(chat_id, message_id)] = {
            "chat_title": chat_title,
            "voice_file_id": msg.voice.file_id
        }
        msg.reply_text("Voice message received. Awaiting caption...")
    else:
        caption = caption.title().replace(" ", "_")
        ogg_file_name = f'{chat_title}_{caption}.ogg'
        print(f"Downloading file: {ogg_file_name}")

        # Download the file
        new_file = context.bot.get_file(update.message.voice.file_id)
        new_file.download(ogg_file_name)

        model, utils = load_vad_model()
        audio = load_audio(ogg_file_name)
        speech_timestamps = detect_speech_intervals(audio, model)
        audio_segment = AudioSegment.from_ogg(ogg_file_name)
        parts = []
        for i, ts in enumerate(speech_timestamps):
            start_ms = int(ts['start'] * 1000 / 16000)
            end_ms = int(ts['end'] * 1000 / 16000)
            segment = audio_segment[start_ms:end_ms]
            parts.append(segment)
            segment.export(f"{chat_title}_{caption}_sentence_{i}.ogg", format="ogg")

        msg.reply_text("Download & Segmentation succeeded!")

        # Convert to MP3
        # mp3_file_name = f'{chat_title}_{caption}.mp3'
        # convert_ogg_to_mp3(ogg_file_name, mp3_file_name)
        # msg.reply_text("Converted to MP3 successfully!")

# checks for text messages that may relpy for the pending voices, indicating their caption
def handle_text(update: Update, context: CallbackContext):
    msg = update.effective_message

    if msg.reply_to_message:
        reply_to = msg.reply_to_message
        chat_id = reply_to.chat.id
        message_id = reply_to.message_id

        if (chat_id, message_id) in pending_voice_messages:
            voice_info = pending_voice_messages.pop((chat_id, message_id))
            chat_title = voice_info["chat_title"]
            voice_file_id = voice_info["voice_file_id"]
            caption = msg.text

            caption = caption.title().replace(" ", "_")
            ogg_file_name = f'{chat_title}_{caption}.ogg'

            print(f"Downloading file: {ogg_file_name}")

            # Download the file
            new_file = context.bot.get_file(voice_file_id)
            new_file.download(ogg_file_name)
            msg.reply_text("Download succeeded!")

            # Convert to MP3
            # mp3_file_name = f'{chat_title}_{caption}.mp3'
            # convert_ogg_to_mp3(ogg_file_name, mp3_file_name)
            # msg.reply_text("Converted to MP3 successfully!")

# handle files, some voices may not be in the telegram voice format so this function stores them directly
def handle_files(update: Update, context: CallbackContext):
    msg = update.effective_message
    attach = update.message.effective_attachment

    file_name = attach.file_name
    print(f"Downloading file: {file_name}")
    new_file = context.bot.get_file(attach.file_id)
    new_file.download(file_name)
    msg.reply_text(f"file '{file_name}' has been downloaded and saved successfully.")

def load_vad_model():
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)
    return model, utils

def load_audio(file_path):
    audio = AudioSegment.from_ogg(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    return np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0

def detect_speech_intervals(audio, model):
    return get_speech_timestamps(audio, model, sampling_rate=16000)


if __name__ == '__main__':
    print("Starting")

    updater = Updater(TOKEN)

    updater.dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice))
    updater.dispatcher.add_handler(MessageHandler(Filters.text, handle_text))
    updater.dispatcher.add_handler(MessageHandler(Filters.attachment, handle_files))

    print("Pooling")
    updater.start_polling()
    updater.idle()
