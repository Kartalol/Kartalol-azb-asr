import pandas as pd
import re
import csv
import re

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
            my_str += ' ' + _words_num[digit]
    return f'{number} :   {my_str}'


def replace_digits_with_string(text):
    # Replace all digits with the specified token
    numbers = re.findall(r'\d', text)
    for number in numbers:
        text = text.replace(number, number_to_text(number))

    return text

if __name__ == '__main__':
    df1 = pd.read_csv('../book1.csv')
    df2 = pd.read_csv('../book2.csv')
    df3 = pd.read_csv('../book3.csv')
    df4 = pd.read_csv('../book4.csv')
    df5 = pd.read_csv('../book5.csv')
    print(f'{len(df1)}  {len(df2)} {len(df3)}  {len(df4)}  {len(df5)}')

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
