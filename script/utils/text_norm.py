# Colab: read JSON manifest -> normalize text -> filter by word count -> save new JSON for training later
# (keeps same schema: {"data":[{"audio_filepath":..., "duration":..., "text":...}, ...]})


import os, json, re, string

# =========================
# CONFIG (edit paths)
# =========================
INPUT_JSON  = "/content/drive/MyDrive/KartalOl Corpus/Research Papers/ASR Paper/dataset/KartalOl_general_voices_0_0_1_training.json"  # change to your path
OUTPUT_JSON = "/content/drive/MyDrive/KartalOl Corpus/Research Papers/ASR Paper/dataset/KartalOl_general_voices_0_0_1_training_normalized.json"   # saved for training later

MIN_WORDS = 5
MAX_WORDS = 100
# =========================
# Normalization utilities
# =========================
_num_words = {
    "بیر": 1, "ایکی": 2, "اۆچ": 3, "دؤرد": 4, "بئش": 5, "آلتی": 6, "یئددی": 7, "سککیز": 8, "دوْققوز": 9,
    "اوْن": 10, "ایگیرمی": 20, "اوْتوز": 30, "قؽرخ": 40, "اللی": 50, "آلتمیش": 60, "یئتمیش": 70,
    "سکسن": 80, "دوْخسان": 90, "یۆز": 100, "مین": 1000, "میلیون": 1000000, "میلیارد": 10000000,
}
def to_english_digits(text):
    # Mapping both Arabic and Persian digits to English 0-9
    arabic_digits = '٠١٢٣٤٥٦٧٨٩'
    persian_digits = '۰۱۲۳۴۵۶۷۸۹'
    english_digits = '0123456789'
    
    # Create combined translation table
    table = str.maketrans(arabic_digits + persian_digits, english_digits * 2)
    return text.translate(table)
_words_num = {str(v): k for k, v in _num_words.items()}

_num_words = {
    "بیر": 1,  # 1
    "ایکی": 2,
    "اۆچ": 3,
    "دؤرد": 4,
    "بئش": 5,
    "آلتی": 6,
    "یئددی": 7,
    "سککیز": 8,
    "دوْققوز": 9,
    "اوْن": 10,  # 10
    "ایگیرمی": 20,  # 20
    "اوْتوز": 30,
    "قؽرخ": 40,
    "اللی": 50,
    "آلتمیش": 60,
    "یئتمیش": 70,
    "سکسن": 80,
    "دوْخسان": 90,
    "یۆز": 100,  # 100
    "مین": 1000,  # 1000
    "میلیون": 1000000,  # 1000,000
    "میلیارد": 10000000,
}
_words_num = {str(y): x for x, y in _num_words.items()}
def word_count(text: str) -> int:
    return 0 if not text else len([w for w in text.split() if w])

def number_to_text(number, text):
    number_str = str(number)  # Handle both string and int input
    size = len(number_str)
    if size > 4:
        return str(number)

    my_str = ''
    for index, digit in enumerate(number_str):
        if digit == '0':
            continue

        if size - index == 4:  # thousands
            if digit == '1':
                my_str += ' ' + "مین"
            else:
                my_str += ' ' + _words_num[digit]
                my_str += ' ' + "مین"
        elif size - index == 3:  # hundreds
            if digit == '1':
                my_str += ' ' + "یۆز"  # Fixed: "مین" → "یۆز"
            else:
                my_str += ' ' + _words_num[digit]
                my_str += ' ' + "یۆز"
        elif size - index == 2:  # tens
            my_str += ' ' + _words_num[digit + '0']
        else:  # ones
            my_str += ' ' + _words_num[digit]

    return my_str.strip()


def replace_digits_with_string(text):
    numbers = re.findall(r'\d+', text)
    for number in numbers:
        text = text.replace(number, number_to_text(number, text))
    return text


def cleaner_final(text: str):
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'«[^»]*»', '', text)
    text = to_english_digits(text)
    text = replace_digits_with_string(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(' +', ' ', text)

    return text

# =========================
# Load -> normalize -> filter -> save
# =========================
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    obj = json.load(f)

rows = obj.get("data", [])
kept = []
stats = {
    "total": len(rows),
    "kept": 0,
    "dropped_short": 0,
    "dropped_long": 0,
    "dropped_empty": 0,
    "dropped_missing_audio": 0,
}

for r in rows:
    rel = r.get("audio_filepath", "")
    dur = r.get("duration", 0.0)
    txt = r.get("text", "")

    if not isinstance(rel, str) or not rel.strip():
        stats["dropped_empty"] += 1
        continue

    if txt is None:
        txt = ""
    txt = str(txt)
    norm = cleaner_final(txt)

    wc = word_count(norm)
    if wc == 0:
        stats["dropped_empty"] += 1
        continue
    if wc < MIN_WORDS:
        stats["dropped_short"] += 1
        continue
    if wc > MAX_WORDS:
        stats["dropped_long"] += 1
        continue
    kept.append({
        "audio_filepath": rel,          # keep relative path like your original JSON
        "duration": float(dur) if dur is not None else 0.0,
        "text": norm,                   # normalized text
    })

stats["kept"] = len(kept)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"data": kept}, f, ensure_ascii=False, indent=2)

print("Saved:", OUTPUT_JSON)
print(stats)
print("Example cleaned row:", kept[0] if kept else "NO ROWS KEPT")
