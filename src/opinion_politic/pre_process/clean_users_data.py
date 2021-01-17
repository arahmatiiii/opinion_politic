'''
clean_users_data is written for clean user tweets
'''
import pandas
import ast
import re
import hazm
import glob
from opinion_politic.pre_process.normalizer import Normalizing_characters



class clean_user:
    '''
    class for clean and create usres tweets
    '''
    def __init__(self):
        self.new_normalizer = Normalizing_characters()
        self.evaluation_user_path = '../../../data/Intermadiate/evaluation_user_big.csv'
        self.crawled_data_path = '../../../data/Intermadiate/crawled_data/'
        self.evaluation_user_data = '../../../dara/Intermadiate/evaluation_user_data_big/'

    def create_user_files(self):
        evaluation_users = pandas.read_csv(self.evaluation_user_path)

        for user_index, user in enumerate(evaluation_users.user):
            user_tweets = []
            user_csv_path = self.crawled_data_path + user + '.csv'
            user_csv = pandas.read_csv(user_csv_path)
            user_csv["length"] = user_csv["reply_to"].apply(lambda x: len(ast.literal_eval(x)) == 0)
            user_csv_noreply = user_csv.loc[user_csv["length"], :].drop(["length"], axis=1)

            for i, item in enumerate(user_csv_noreply.tweet):
                item = str(item)
                url = re.findall(r"http\S+", item)
                if (url == []):
                    item = self.new_normalizer.Normalizer_text(item)
                    if len(hazm.word_tokenize(item)) >= 5:
                        user_tweets.append(item)

                if i % 1000 == 0:
                    print(f'{(i/len(user_csv_noreply))*100 :.2f} done {user} {user_index}')

            user_csv_noreply_normed = pandas.DataFrame({'tweet': user_tweets})
            user_csv_noreply_normed.to_csv(user_csv_path.replace('crawled_data', 'evaluation_user_data_big'), index=False)

    def delete_empty_user(self):
        evaluation_user_big = pandas.read_csv('../../../data/Intermadiate/evaluation_user_big.csv')
        new_user_username = []
        new_user_laebl = []
        for i, item in enumerate(evaluation_user_big.user):
            user_csv = pandas.read_csv('../../../data/Intermadiate/evaluation_user_data_big/' + item.strip() + '.csv')
            if len(user_csv) != 0:
                new_user_username.append(item)
                new_user_laebl.append(evaluation_user_big.label[i])
        new_evaluation_user_big = pandas.DataFrame({'user': new_user_username, 'label': new_user_laebl})
        new_evaluation_user_big.to_csv('../../../data/Intermadiate/new_evaluation_user_big.csv', index=False)
        print(new_evaluation_user_big.label.value_counts())


if __name__ == '__main__':
    clean_user_object = clean_user()
    clean_user_object.create_user_files()
