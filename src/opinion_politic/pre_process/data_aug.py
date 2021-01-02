import pandas
import gensim
import hazm
import random
import math
import time


def find_sim_w2v(model, word):
    try:
        simi = model.most_similar([word])
        sim_words = []
        for i, item in enumerate(simi):
            sim_words.append(item[0])

        selected = random.sample(sim_words, k=1)[0]
    except:
        selected = word

    return selected


def delete_random(item_tokenized):
    count = math.floor(((25 * len(item_tokenized)) / 100))
    while count > 0:
        random_index = random.randint(0, len(item_tokenized) - 1)
        random_word = item_tokenized[random_index]
        item_tokenized.remove(random_word)
        count = count - 1

    result = ' '.join(item_tokenized)
    return result


def data_w2v_aug(num_aug, item_tokenized, w2v_model):
    count = math.floor((((num_aug + 1) * 20 * len(item_tokenized)) / 100))
    while count > 0:
        random_index = random.randint(0, len(item_tokenized) - 1)
        random_word = item_tokenized[random_index]
        translated = find_sim_w2v(w2v_model, random_word)
        item_tokenized[random_index] = translated
        count = count - 1

    result = ' '.join(item_tokenized)
    return result


def data_aug(dataset, w2c_model):
    train_aug_text = []
    train_aug_label = []

    start_time = time.time()
    for i, item in enumerate(dataset.text):
        item = str(item)
        item_label = dataset.label[i]
        train_aug_text.append(item)
        train_aug_label.append(item_label)

        item_tokenized = hazm.word_tokenize(item)
        for num_aug in range(2):
            result = data_w2v_aug(num_aug, item_tokenized, w2c_model)
            train_aug_text.append(result)
            train_aug_label.append(item_label)


        if i % 1000 == 0:
            print(f'{i} ta rafte {time.time() - start_time}')
            start_time = time.time()

    auged_dataframe = pandas.DataFrame({'text': train_aug_text, 'label': train_aug_label})
    return auged_dataframe


if __name__ == '__main__':

    data_train_path = './data/train_data_normed.csv'
    w2v_path = './data/skipgram_news_300d_30e.txt'

    dataset = pandas.read_csv(data_train_path)
    print('dataset opend')
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
    print('w2v model loaded')
    print('data aug started')
    auged_dataframe = data_aug(dataset=dataset, w2c_model=w2v_model)

    print(dataset.label.value_counts())
    print(auged_dataframe.label.value_counts())

    auged_dataframe.to_csv(data_train_path.replace('.csv', '_augged_new.csv'))

