
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch
import torchaudio

# Load model and processor
model_id = "facebook/mms-1b-all"
fprocessor = AutoProcessor.from_pretrained(model_id)
fmodel = Wav2Vec2ForCTC.from_pretrained(model_id)

fprocessor.tokenizer.set_target_lang("azb")
fmodel.load_adapter("azb")
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor
import torch
import librosa
# psth = "/content/drive/MyDrive/KartalOl Corpus/Dataset/Speech Recognition dataset/Work with Farhan/code/fine_tune_model/"
# model = AutoModelForCTC.from_pretrained(psth+"checkpoint-1200").to("cuda")
# processor = Wav2Vec2BertProcessor.from_pretrained(psth)
def infer(file: str):
    audio, sr = librosa.load(
        file,
        sr=16000,
    )
    input_values = fprocessor(audio, sampling_rate=sr, return_tensors="pt").to(
        fmodel.device
    )
    # ins = torch.tensor(input_values["input_features"]).to("cuda").unsqueeze(0)
    logits = fmodel(**input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)[0]

    return fprocessor.decode(pred_ids)


infer("/home/amber/Desktop/KartalOl/code/Kartalol-speech-recognition/dataset/kartalol_gold_testset/voices/1-@Danulduzuyam-audio_2023-11-27_13-56-10.ogg")