'''
words_analysis.py is written for analysis words
'''


import pandas
from collections import Counter
import hazm


def load_data(dataset_path):
    train_dataset = pandas.read_csv(dataset_path)
    pos_text = []
    nex_text = []
    for i, item in enumerate(train_dataset.label):
        if item == 1:
            pos_text.append(train_dataset.text[i])
        else:
            nex_text.append(train_dataset.text[i])

    return pos_text, nex_text


def find_mostfreq(sentence_list, n):
    counter = Counter()
    for item in sentence_list:
        counter.update(hazm.word_tokenize(str(item)))
    most_occur = counter.most_common(n=n)
    return most_occur


def find_unique_words(pos_most_occer, neg_most_occer):
    pos_most_words = []
    neg_most_words = []
    for item in pos_most_occer:
        pos_most_words.append(item[0])

    for item in neg_most_occer:
        neg_most_words.append(item[0])

    unique_words = pos_most_words + neg_most_words
    unique_words = set(unique_words)
    return unique_words


if __name__ == '__main__':
    pos_text, nex_text = load_data('../../../data/Processed/train_data_normed.csv')
    pos_most_occer = find_mostfreq(pos_text, n=100)
    neg_most_occer = find_mostfreq(nex_text, n=100)

    unique_words = find_unique_words(pos_most_occer=pos_most_occer, neg_most_occer=neg_most_occer)
    print(len(unique_words))
    print(unique_words)
    data_out = pandas.DataFrame({'words': list(unique_words)})
    data_out.to_excel('../../../data/Processed/stopwords.xlsx', index=False)
