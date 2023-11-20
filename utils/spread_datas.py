import pandas as pd
import csv

Females = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
Males = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']
Others = ['o1', 'o2']

Female_data = [{'age': 40, 'device': 'samsung A10 '}, {'age': 19, 'device': 'samsung A20s '},
               {'age': 34, 'device': 'samsung A13 '}, {'age': 19, 'device': 'xiamomi poco x3 '},
               {'age': 22, 'device': 'samsung A9 '}, {'age': 22, 'device': 'xiamomi poco m3'}]

Speaker = Females + Males + Others


def save_csv(df):
    speakers = df.groupby(['ID'])
    for i in range(1, 3):
        speakers.get_group(f'o{i}').to_csv(f'o{i}.csv', sep='\t', index=False, header=True)


def avarage_sent_length(df):
    avarage_list = []
    for i in range(len(df['Sentence'])):
        avarage_list.append(len(df['Sentence'][i].split()))
    print(sum(avarage_list) / len(avarage_list))


def df_to_csv(df):
    return df


def all_to_text(df):
    full_text = ''
    for i in range(len(df['Sentence'])):
        full_text += ' ' + df['Sentence'][i]
    return len(set(full_text.split()))


def create_csv(lists):
    rows = list()

    for i in range(len(lists)):
        row = [i + 1, lists[i]['Sentence'], lists[i]['ID'], lists[i]['book']]
        rows.append(row)

    header = ['NO', 'Sentence', 'ID', 'book_name']

    with open('../../all_test_not_real.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == '__main__':
    df1 = pd.read_csv('../../book1.csv')
    df2 = pd.read_csv('../../book2.csv')
    df3 = pd.read_csv('../../book3.csv')
    df4 = pd.read_csv('../../book4.csv')
    df5 = pd.read_csv('../../book5.csv')
    # print(df1['Sentence'])
    my_list = list()
    #
    for i in range(len(df1)):
        data = {'ID': Speaker[i % len(Speaker)], 'Sentence': df1['Sentence'][i], 'book': 'book1'}
        my_list.append(data)
    for i in range(len(df2)):
        data = {'ID': Speaker[i % len(Speaker)], 'Sentence': df2['Sentence'][i], 'book': 'book2'}
        my_list.append(data)
    for i in range(len(df3)):
        data = {'ID': Speaker[i % len(Speaker)], 'Sentence': df3['Sentence'][i], 'book': 'book3'}
        my_list.append(data)
    for i in range(len(df4)):
        data = {'ID': Speaker[i % len(Speaker)], 'Sentence': df4['Sentence'][i], 'book': 'book4'}
        my_list.append(data)
    for i in range(len(df5)):
        data = {'ID': Speaker[i % len(Speaker)], 'Sentence': df5['Sentence'][i], 'book': 'book5'}
        my_list.append(data)

    create_csv(my_list)

    df = pd.read_csv('../../all_test_not_real.csv')
    print(all_to_text(df))
    save_csv(df)
    df1 = pd.read_csv("../datasets/sentences/f1.csv", delimiter='\t', header=None, skiprows=1,
                      names=['NO', 'Sentence', 'ID', 'book_name'])
    df2 = pd.read_csv("../datasets/sentences/f2.csv", delimiter='\t', header=None, skiprows=1,
                      names=['NO', 'Sentence', 'ID', 'book_name'])
    df3 = pd.read_csv("../datasets/sentences/f3.csv", delimiter='\t', header=None, skiprows=1,
                      names=['NO', 'Sentence', 'ID', 'book_name'])
    df4 = pd.read_csv("../datasets/sentences/f4.csv", delimiter='\t', header=None, skiprows=1,
                      names=['NO', 'Sentence', 'ID', 'book_name'])
    df5 = pd.read_csv("../datasets/sentences/f5.csv", delimiter='\t', header=None, skiprows=1,
                      names=['NO', 'Sentence', 'ID', 'book_name'])
    df6 = pd.read_csv("../datasets/sentences/f6.csv", delimiter='\t', header=None, skiprows=1,
                      names=['NO', 'Sentence', 'ID', 'book_name'])
    dfm1 = pd.read_csv("../datasets/sentences/m1.csv", delimiter='\t', header=None, skiprows=1,
                       names=['NO', 'Sentence', 'ID', 'book_name'])
    dfm2 = pd.read_csv("../datasets/sentences/m2.csv", delimiter='\t', header=None, skiprows=1,
                       names=['NO', 'Sentence', 'ID', 'book_name'])
    dfm3 = pd.read_csv("../datasets/sentences/m3.csv", delimiter='\t', header=None, skiprows=1,
                       names=['NO', 'Sentence', 'ID', 'book_name'])
    dfm4 = pd.read_csv("../datasets/sentences/m4.csv", delimiter='\t', header=None, skiprows=1,
                       names=['NO', 'Sentence', 'ID', 'book_name'])
    dfm5 = pd.read_csv("../datasets/sentences/m5.csv", delimiter='\t', header=None, skiprows=1,
                       names=['NO', 'Sentence', 'ID', 'book_name'])
    dfm6 = pd.read_csv("../datasets/sentences/m6.csv", delimiter='\t', header=None, skiprows=1,
                       names=['NO', 'Sentence', 'ID', 'book_name'])
    # save_csv(df)

    print('#uniuqe words')
    print(all_to_text(df1))
    print(all_to_text(df2))
    print(all_to_text(df3))
    print(all_to_text(df4))
    print(all_to_text(df5))
    print(all_to_text(df6))

    print(all_to_text(dfm1))
    print(all_to_text(dfm2))
    print(all_to_text(dfm3))
    print(all_to_text(dfm4))
    print(all_to_text(dfm5))
    print(all_to_text(dfm6))

    print()
    print('#avarage words')
    avarage_sent_length(df)
    avarage_sent_length(df1)
    avarage_sent_length(df2)
    avarage_sent_length(df3)
    avarage_sent_length(df4)
    avarage_sent_length(df5)
    avarage_sent_length(df6)

    avarage_sent_length(dfm1)
    avarage_sent_length(dfm2)
    avarage_sent_length(dfm3)
    avarage_sent_length(dfm4)
    avarage_sent_length(dfm5)
    avarage_sent_length(dfm6)
