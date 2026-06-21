import pandas as pd
import re
import csv
import re
from digit_conversion import replace_numbers_in_text 

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


def count_words(sentence):
    number_of_words = len(re.findall(r'\w+', sentence))
    return number_of_words


def words_in_paranthesis(sent):
    return re.findall(r'\([^)]*\)', sent)

def have_speciall(text):
    if '٪' in text or '$' in text or '+' in text:
        print(text)


def text_seperator(text):
    text = text.replace('و...', '.')

    text = text.replace('.', '.SEPERATOR').replace('؟', ' ?SEPERATOR ').replace('!', '!SEPERATOR').replace(':',
                                                                                                           ':SEPERATOR')
    text = text.replace('?', '؟')
    texts = re.split('SEPERATOR', text)
    return texts


def create_csv(QA):
    rows = list()

    for i in range(len(QA)):
        row = [i + 1, QA[i], 'book5']
        rows.append(row)

    header = ['NO', 'Sentence', 'book_name']

    with open('../book5.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def number_to_text(number):
    size = len(number)
    if size > 4:
        return number
    my_str = ''
    for index, digit in enumerate(number):
        if digit == '0':
            continue
        if size - index == 4:
            if digit == '1':
                my_str += ' ' + "مین"
            else:
                my_str += ' ' + _words_num[digit]
                my_str += ' ' + "مین"
        elif size - index == 3:
            if digit == '1':
                my_str += ' ' + "مین"
            else:
                my_str += ' ' + _words_num[digit]
                my_str += ' ' + "یۆز"
        elif size - index == 2:
            my_str += ' ' + _words_num[digit + '0']
        else:
            try:
                my_str += ' ' + _words_num[digit]
            except:
                breakpoint()
    return f'{number} :   {my_str}'


def replace_digits_with_string(text):
    # Replace all digits with the specified token
    numbers = re.findall(r'\d', text)
    for number in numbers:
        text = text.replace(number, number_to_text(number))

    return text
from datasets import Dataset, DatasetDict

import os
import re
import unicodedata

def clean_speech_text(text):
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)  # Remove punctuation except Persian/Arabic script
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.replace("\u200c", "").replace("\xa0", "")
    text = re.sub(r'[.,،؛;!?/]', '', text)

    return text

import librosa
if __name__ == '__main__':
    path = "/home/omniaz/Desktop/2023/projects/Kartalol-speech-recognition/dataset/Kartal Ol Gold Test set/"
    data = pd.read_excel(path+'sentences.xlsx')
    voices_list = os.listdir(path+"voices")
    # Ensure that ids in filenames match the Excel file
    audio_files = [
            os.path.join(path+"voices", f)
            for f in os.listdir(path+"voices")
            if f.endswith(".ogg")
        ]
    dataset_entries = []
    # Ensure that ids in filenames match the Excel file
    male = [
        "jalil", 
        "hasan", 
        "Application_support", 
        "qhashimi",
        "javad",
        "Aamin1989",
        "aamin",
        "beheshti",
        "m=qhashimi",
        "akbar",
        "Avatar",
        "hojatmosavi",
        "morteza1376",
        "mehdi",
        "mhakan",
        "behnam",
        "Aamin1989",
        "dadrasarsalani",
        "ilki",
        "dadarasaslani",
        "pasha",
        "turk_termili",
        "elqami44",
        "avatar",
        "Turk_tarmili",
        "ramin_g136075"

        ]

    for f in audio_files:
        file_id = os.path.splitext(os.path.basename(f))[0].split("-")[0]
        try:
            if int(file_id) in data["id"]:
                index_list = data.index[data['id'] == int(file_id)].tolist()
                sentence = data["azb"].iloc[index_list].values[0]
                y, sr = librosa.load(f, sr=None)  # Load with original sampling rate

                # Calculate duration
                duration = librosa.get_duration(y=y, sr=sr)
                
                r_f = replace_numbers_in_text(sentence)
                c_word = count_words(r_f)
                text = clean_speech_text(r_f)
                if len(f.split("/")[-1].split("-")[1]) == 1:
                    gen = f.split("/")[-1].split("-")[1]
                elif f.split("/")[-1].split("-")[1] in male:
                    gen = "m"
                else:
                    gen = "f"
                    print(f.split("/")[-1].split("-")[1])
                dataset_entries.append({
                "audio_filepath": "voices/"+f.split("/")[-1],
                "labels": text,
                "count_words": c_word,
                "voice_duration": duration,
                "gender": gen,
            })
        except:
            breakpoint()

    # Convert to a Hugging Face Dataset
    full_dataset = Dataset.from_list(dataset_entries)
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "test": full_dataset
    })
    dataset_dict.save_to_disk("ASR-AZB-Gold-TestSet")
    from datasets import DatasetDict

    # Save as Parquet
    dataset_dict.save_to_disk("ASR-AZB-Gold-TestSet")
    dataset_dict["test"].to_parquet("ASR-AZB-Gold-TestSet.parquet")


    # full_txt = ' '
    # with open('book5.txt') as f:
    #     for line in f.readlines():
    #         if not line.isspace():
    #             full_txt += ' ' + line + ' '
    #
    # # for some states like باشه...
    # full_txt = re.sub(r'\([^)]*\)', ' ', full_txt)
    # full_txt = re.sub(r'«[^»]*»', '', full_txt)
    # full_txt = replace_digits_with_string(full_txt)
    # my_list = text_seperator(full_txt)
    #
    # sent_list = []
    # filtered_sentence = [re.sub(' +', ' ', item.strip().replace('\n', ' ')) for item in my_list if
    #                      item.strip() != "" and count_words(item) > 5]
    #
    # for sent in filtered_sentence:
    #     sent_list.append(sent)

# df = pd.read_csv('book4.csv')
# counter = 0
# sent_list = list()
# for i in range(len(df['Sentence'])):
#     words_in_paranthesis(df['Sentence'][i])
#     if re.findall(r'\d+', df['Sentence'][i]):
#         print(re.findall(r'\d+', df['Sentence'][i]))
# sent_list.append(re.sub(r'\([^)]*\)', ' ', df['Sentence'][i]))
# if words_in_paranthesis(df['Sentence'][i]):
#     print(words_in_paranthesis(df['Sentence'][i]))
# print(sent_list)
# create_csv(sent_list)
