'''
ml_run.py is written for machine learning models
'''

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import pandas
import pickle
from opinion_politic.config.apcnn_lstm_config import \
    TRAIN_DATA_PATH, TEST_DATA_PATH, VALID_DATA_PATH


class ML_MODELS():

    def __init__(self):
        self.train_text, self.train_label = self.load_datasets(TRAIN_DATA_PATH)
        self.test_text, self.test_label = self.load_datasets(TEST_DATA_PATH)
        self.valid_text, self.valid_label = self.load_datasets(VALID_DATA_PATH)

    def load_datasets(self, dataset_path):
        dataset = pandas.read_csv(dataset_path)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset.astype({'text': 'str'})
        dataset = dataset.astype({'label': 'int'})
        dataset_text = list(dataset.text)
        dataset_label = list(dataset.label)
        return dataset_text, dataset_label

    def nb_model(self):
        text_clf_NB = Pipeline([("vect", TfidfVectorizer()),
                                ("clf", MultinomialNB())])
        text_clf_NB.fit(self.train_text, self.train_label)
        return text_clf_NB

    def svm_model(self):
        text_clf_SVM = Pipeline([("vect", TfidfVectorizer()),
                                ("clf", SGDClassifier())])
        text_clf_SVM.fit(self.train_text, self.train_label)
        return text_clf_SVM

    def predict_data(self, model, data_text, data_label):
        predict_label = model.predict(data_text)
        print(classification_report(y_true=data_label, y_pred=predict_label))

    def create_model_eval(self, model):
        if model == 'nb':
            trained_model = self.nb_model()
        elif model == 'svm':
            trained_model = self.svm_model()

        print('train eval')
        self.predict_data(model=trained_model, data_text=self.train_text, data_label=self.train_label)
        print('test eval')
        self.predict_data(model=trained_model, data_text=self.test_text, data_label=self.test_label)
        print('valid eval')
        self.predict_data(model=trained_model, data_text=self.valid_text, data_label=self.valid_label)


if __name__ == '__main__':
    ml_models_pbj = ML_MODELS()
    ml_models_pbj.create_model_eval('nb')
    # ml_models_pbj.create_model_eval('svm')




