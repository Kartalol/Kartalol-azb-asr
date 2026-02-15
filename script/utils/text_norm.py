import csv
import re, string
from parsinorm import General_normalization

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


def number_to_text(number):
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
        text = text.replace(number, number_to_text(number))
    return text


general_normalization = General_normalization()


def cleaner_final(text: str):
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'«[^»]*»', '', text)
    text = replace_digits_with_string(text)
    general_normalization.alphabet_correction(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(' +', ' ', text)

    return text


if name == '__main__':
    your_text = ''
    print(cleaner_final(your_text))
