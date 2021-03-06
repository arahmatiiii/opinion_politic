"""
lstm_test.py is written for test LSTM model
"""

import torch
import hazm
import pandas
import random
import math
import gensim
import logging
from collections import Counter
from sklearn.metrics import classification_report
from opinion_politic.methods.lstm_model import LSTM
from opinion_politic.evaluation.lstm_evaluation import PREDICT_SAMPLE_LSTM
from opinion_politic.config.lstm_config import TEST_DATA_PATH,\
    START_DROPOUT, MIDDLE_DROPOUT, FINAL_DROPOUT,\
    MODEL_PATH, DEVICE, EVAL_USER_DATA_PATH, EVAL_USER_PATH, \
    TEXT_FIELD_PATH, EMBEDDING_DIM, HIDDEN_DIM, BIDIRECTIONAL, N_LAYERS, EMBEDDING_PATH

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class TestModel:
    '''
    class for load and predict sample
    '''
    def __init__(self):

        self.text_field = torch.load(TEXT_FIELD_PATH)
        self.tokenizer = hazm.word_tokenize
        self.model_path = MODEL_PATH

    def init_model(self):
        """
        init_model method is written for loading model and
        define loss function and optimizer
        :param data_set:
        :param use_pos:
        :return:
            criterion: loss function
            optimizer: optimizer function
        """
        # create model
        model = LSTM(vocab_size=len(self.text_field.vocab),
                     embedding_dim=EMBEDDING_DIM,
                     pad_idx=self.text_field.vocab.stoi[self.text_field.pad_token],
                     hidden_dim=HIDDEN_DIM,
                     n_layers=N_LAYERS,
                     bidirectional=BIDIRECTIONAL,
                     start_dropout=START_DROPOUT,
                     middle_dropout=MIDDLE_DROPOUT,
                     final_dropout=FINAL_DROPOUT,
                     output_size=2)

        return model


    def test(self, test_sentence):
        """
        run method is written for running model
        """
        model = self.init_model()
        model.load_state_dict(torch.load(self.model_path + 'final_model.pt', map_location=DEVICE), strict=False)

        predict_sample = PREDICT_SAMPLE_LSTM(model=model.to(DEVICE), tokenizer=self.tokenizer,
                                             text_field=self.text_field)
        results_tagged = predict_sample.predict_lstm(test_sentence)

        return results_tagged


class TestUser:
    '''
    class for uvaluate on classification users
    '''
    def __init__(self):
        self.eval_user_path = EVAL_USER_PATH
        self.evaluation_users = pandas.read_csv(self.eval_user_path)
        self.test_lstm_model = TestModel()
        self.eval_user_data_path = EVAL_USER_DATA_PATH

    def predict_user(self):

        all_eval_users_username = list(self.evaluation_users.user)
        all_eval_users_label = list(self.evaluation_users.label)
        all_eval_usres_pred_prop = []
        all_eval_usres_pred_label = []

        for user_index, user in enumerate(self.evaluation_users.user):
            user_pos_count = 0
            user_neg_count = 0
            user_csv_file = pandas.read_csv(self.eval_user_data_path + user + '.csv')
            len_user_csv_file = len(user_csv_file)
            for i, item in enumerate(user_csv_file.tweet):
                if self.test_lstm_model.test(item) == 1:
                    user_pos_count += 1
                else:
                    user_neg_count += 1

                if i % 1000 == 0:
                    print(f'{(i/len_user_csv_file)*100 :.2f} done {user} {user_index}')

            pos_propb = user_pos_count/len_user_csv_file * 100
            neg_prob = user_neg_count/len_user_csv_file * 100

            if pos_propb >= neg_prob:
                user_pred_label = 1
            else:
                user_pred_label = 0

            all_eval_usres_pred_prop.append([pos_propb, neg_prob])
            all_eval_usres_pred_label.append(user_pred_label)

        all_results = pandas.DataFrame({'user': all_eval_users_username,
                                        'label': all_eval_users_label,
                                        'pred_label': all_eval_usres_pred_label,
                                        'pred_prob': all_eval_usres_pred_prop})

        all_results.to_csv(self.eval_user_path.replace('.csv', '_results_lstm.csv'), index=False)

    def report_metric_user(self):

        evaluation_user_res = pandas.read_csv(self.eval_user_path.replace('.csv', '_results_lstm.csv'))

        main_label = list(evaluation_user_res.label)
        pred_label = list(evaluation_user_res.pred_label)

        print(classification_report(y_true=main_label, y_pred=pred_label))



if __name__ == '__main__':

    test_user_obj = TestUser()
    test_user_obj.predict_user()
    test_user_obj.report_metric_user()
