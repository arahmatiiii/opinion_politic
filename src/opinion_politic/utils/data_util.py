"""
data_util.py is writen for creating iterator and save field
"""

import logging
import pandas
import torch
import hazm
from torchtext import data
from torchtext.vocab import Vectors
import numpy as np
from sklearn.utils import class_weight
from opinion_politic.config.apcnn_lstm_config import TEXT_FIELD_PATH,\
    LABEL_FIELD_PATH, BATCH_SIZE, DEVICE, USE_STOPWORD, STOPWORDS_PATH, \
    IS_TRANSFORMER, IS_ELMO, MAX_LEN


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class DataSet:
    """
    DataSet Class use for preparing data
    and iterator for training model.
    """
    def __init__(self, **kwargs):
        self.files_address = {"train_data_path": kwargs["train_data_path"],
                              "test_data_path": kwargs["test_data_path"],
                              "valid_data_path": kwargs["valid_data_path"],
                              "embedding_path": kwargs["embedding_path"]}

        self.iterator_dict = dict()
        self.class_weight = None
        self.embedding_dict = {'vocab_embedding_vector':None}
        self.unk_idx_dict = {"token_unk_idx": None}
        self.pad_idx_dict = {"token_pad_idx": None}
        self.num_vocab_dict = {"num_token": 0, 'num_label': 0}
        self.text_field = None
        self.dictionary_fields = dict()

    @staticmethod
    def read_input_file(input_addr):
        """
        Reading input csv file and calculate class_weight
        :return: dataFrame
        """
        dataset = pandas.read_csv(input_addr)
        dataset = pandas.DataFrame(dataset)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        dataset = dataset.astype({"text": "str"})
        dataset = dataset.astype({"label": "int"})

        print('value count in label is : \n', dataset['label'].value_counts())
        return dataset[:100]

    @staticmethod
    def create_stopword():
        stopwords_data = pandas.read_excel(STOPWORDS_PATH)
        return list(stopwords_data.words)

    def create_fields(self):
        """
        This method is writen for creating torchtext fields
        :return: dictionary_fields, datafields
        """
        # Create Field for data
        if USE_STOPWORD:
            stop_word = self.create_stopword()
            text_field = data.Field(sequential=True, tokenize=hazm.word_tokenize, batch_first=True,
                                    include_lengths=True, stop_words=stop_word, fix_length=MAX_LEN)
        else:
            text_field = data.Field(sequential=True, tokenize=hazm.word_tokenize, batch_first=True,
                                    include_lengths=True, fix_length=MAX_LEN)

        if IS_TRANSFORMER:
            text_field = data.Field(sequential=True, tokenize=hazm.word_tokenize, batch_first=True,
                                    include_lengths=True, init_token='<sos>', eos_token='<eos>', fix_length=MAX_LEN)

        label_field = data.LabelField()

        dictionary_fields = {"text_field": text_field, "label_field": label_field}
        datafields = [("text", text_field), ("label", label_field)]

        return dictionary_fields, datafields

    def load_data(self):
        """
        Create iterator for train and test data
        """
        # create fields
        logging.info("Start creating fields.")
        self.dictionary_fields, data_fields = self.create_fields()

        logging.info("Start creating train example.")
        train_examples = [data.Example.fromlist(i, data_fields) for i in
                          self.read_input_file(self.files_address["train_data_path"]).values.tolist()]
        train_data = data.Dataset(train_examples, data_fields)

        logging.info("Start creating test example.")
        test_examples = [data.Example.fromlist(i, data_fields) for i in
                         self.read_input_file(self.files_address["test_data_path"]).values.tolist()]
        test_data = data.Dataset(test_examples, data_fields)

        logging.info("Start creating valid example.")
        valid_examples = [data.Example.fromlist(i, data_fields) for i in
                         self.read_input_file(self.files_address["valid_data_path"]).values.tolist()]
        valid_data = data.Dataset(valid_examples, data_fields)

        # build vocab in all fields
        if IS_TRANSFORMER or IS_ELMO:
            logging.info("Start creating text_field vocabs.")
            self.dictionary_fields["text_field"].build_vocab(train_data,
                                                             min_freq=10)

        else:
            logging.info("Start creating text_field vocabs.")
            self.dictionary_fields["text_field"].build_vocab(train_data,
                                                             min_freq=10,
                                                             unk_init=torch.Tensor.normal_,
                                                             vectors=Vectors(self.files_address["embedding_path"]))

        self.text_field = self.dictionary_fields["text_field"]

        if (not IS_TRANSFORMER) or (not IS_ELMO):
            # get embedding vector for all vocabs
            self.embedding_dict["vocab_embedding_vectors"] = \
                self.dictionary_fields["text_field"].vocab.vectors

        logging.info("Start creating label_field vocabs.")
        self.dictionary_fields["label_field"].build_vocab(train_data)

        print(self.dictionary_fields["label_field"].vocab.stoi)

        # count number of unique vocab in all fields
        self.num_vocab_dict = self.calculate_num_vocabs(self.dictionary_fields)

        # get pad index in all fields
        self.pad_idx_dict = self.find_pad_index(self.dictionary_fields)

        # get unk index
        self.unk_idx_dict = self.find_unk_index(self.dictionary_fields)

        # calculate class weight for handling imbalanced data
        self.class_weight = self.calculate_class_weight(self.dictionary_fields)

        # saving fields
        logging.info("Start saving fields...")
        self.save_feilds(self.dictionary_fields)

        # creating iterators for training model
        logging.info("Start creating iterator.")
        self.iterator_dict = self.creating_iterator(train_data=train_data,
                                                    dev_data=valid_data,
                                                    test_data=test_data)

        logging.info("Loaded %d train examples", len(train_data))
        logging.info("Loaded %d validation examples", len(valid_data))
        logging.info("Loaded %d test examples", len(test_data))

    @staticmethod
    def save_feilds(dictionary_fields):
        """
        This method is writen for saving fields
        :param dictionary_fields: dictionary of fields
        """
        torch.save(dictionary_fields["text_field"], TEXT_FIELD_PATH)
        logging.info("text_field is saved.")

        torch.save(dictionary_fields["label_field"], LABEL_FIELD_PATH)
        logging.info("label_field is saved.")

    @staticmethod
    def calculate_class_weight(dictionary_fields):
        """
        This method calculate class weight
        :param dictionary_fields: dictionary of fields
        :return:
            class_weights: calculated class weight
        """
        label_list = []
        for label, idx in dictionary_fields["label_field"].vocab.stoi.items():
            for _ in range(dictionary_fields["label_field"].vocab.freqs[label]):
                label_list.append(idx)
        class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                          classes=np.unique(label_list),
                                                          y=label_list).astype(np.float32)
        return class_weights

    @staticmethod
    def creating_iterator(**kwargs):
        """
        This method create iterator for training model
        :param kwargs:
            train_data: train dataSet
            valid_data: validation dataSet
            test_data: test dataSet
            human_test_data: human test dataSet
        :return:
            iterator_dict: dictionary of iterators
        """
        iterator_dict = {"train_iterator": data.BucketIterator(kwargs["train_data"],
                                                               batch_size=BATCH_SIZE,
                                                               sort=False,
                                                               shuffle=True,
                                                               device=DEVICE),
                         "dev_iterator": data.BucketIterator(kwargs["dev_data"],
                                                             batch_size=BATCH_SIZE,
                                                             sort=False,
                                                             shuffle=True,
                                                             device=DEVICE),
                         "test_iterator": data.BucketIterator(kwargs["test_data"],
                                                              batch_size=BATCH_SIZE,
                                                              sort=False,
                                                              shuffle=True,
                                                              device=DEVICE)}
        return iterator_dict

    @staticmethod
    def calculate_num_vocabs(dictionary_fields):
        """
        This method calculate vocab counts in each field
        :param dictionary_fields: dictionary of fields
        :return:
            num_vocab_dict:  dictionary of vocab counts in each field
        """
        num_vocab_dict = dict()
        num_vocab_dict["num_token"] = len(dictionary_fields["text_field"].vocab)
        num_vocab_dict["num_label"] = len(dictionary_fields["label_field"].vocab)
        return num_vocab_dict

    @staticmethod
    def find_pad_index(dictionary_fields):
        """
        This method find pad index in each field
        :param dictionary_fields: dictionary of fields
        :return:
            pad_idx_dict: dictionary of pad index in each field
        """
        pad_idx_dict = dict()
        pad_idx_dict["token_pad_idx"] = dictionary_fields["text_field"]\
            .vocab.stoi[dictionary_fields["text_field"].pad_token]

        return pad_idx_dict

    @staticmethod
    def find_unk_index(dictionary_fields):
        """
        This method find unk index in each field
        :param dictionary_fields: dictionary of fields
        :return:
            unk_idx_dict: dictionary of unk index in each field
        """
        unk_idx_dict = dict()
        unk_idx_dict["token_unk_idx"] = dictionary_fields["text_field"] \
            .vocab.stoi[dictionary_fields["text_field"].unk_token]
        return unk_idx_dict


def init_weights(model):
    """
    This method initialize model parameters
    :param model: input model
    """
    for _, param in model.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.1)


def initialize_weights_xavier_uniform_(model):
    """
    This method initialize model parameters
    :param model: input model
    """
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(model.weight.data)

# if __name__ == '__main__':
#     DATA_SET = DataSet(train_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH,
#                        valid_data_path=VALID_DATA_PATH, embedding_path=EMBEDDING_PATH)
#     DATA_SET.load_data()
