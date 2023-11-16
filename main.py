import pandas as pd
import re
import csv
import re


def count_words(sentence):
    number_of_words = len(re.findall(r'\w+', sentence))
    return number_of_words


def text_seperator(text):
    text = text.replace('...', '.')

    text = text.replace('.', '.SEPERATOR').replace('؟', 'SEPERATOR؟').replace('!', '!SEPERATOR').replace(':',
                                                                                                         ':SEPERATOR')
    texts = re.split('SEPERATOR', text)
    return texts


def create_csv(QA):
    rows = list()

    for i in range(len(QA)):
        row = [i, QA[i], 'book2']
        rows.append(row)

    header = ['NO', 'Sentence', 'book_name']

    with open('book2.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == '__main__':
    full_txt = ' '
    with open('book2.txt') as f:
        for line in f.readlines():
            if not line.isspace():
                full_txt += ' ' + line + ' '

    ## for some states like باشه...
    my_list = text_seperator(full_txt)

    sent_list = []
    filtered_sentence = [re.sub(' +', ' ', item.strip().replace('\n', ' ')) for item in my_list if
                         item.strip() != "" and count_words(item) > 5]

    for sent in filtered_sentence:
        sent_list.append(sent)
    create_csv(sent_list)
