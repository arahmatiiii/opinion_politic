"""
elmo_lstm_run.py is written for run elmo_lstm model
"""

import time
import logging
from torch import optim
from torch import nn
import torch
from opinion_politic.utils.augmentation import Augmentation
from opinion_politic.utils.data_util import DataSet, init_weights
from opinion_politic.utils.log_helper import count_parameters, process_time, \
    model_result_log, model_result_save, draw_curves, test_aug_result_log
from opinion_politic.train.train import train, evaluate, evaluate_aug_text

from opinion_politic.methods.elmo_lstm_model import ELMo
from opinion_politic.config.elmo_lstm_config import LOG_PATH, \
    TRAIN_DATA_PATH, TEST_DATA_PATH, VALID_DATA_PATH, \
    EMBEDDING_PATH, EMBEDDING_DIM, START_DROPOUT, \
    MIDDLE_DROPOUT, FINAL_DROPOUT, USE_POS, DEVICE, \
    N_EPOCHS, MODEL_PATH, ELMO_MODEL_PATH, BATCH_SIZE, \
    HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, USE_AUG, \
    TEST_USE_AUG, TEST_AUG_LOG_PATH, STEP_LR, OUTPUT_DIM, MAX_LEN


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


class RunModel:
    """
    In this class we start training and testing model
    """
    def __init__(self):
        # open log file
        self.log_file = open(LOG_PATH, "w")

    @staticmethod
    def load_data_set():
        """
        load_data_set method is written for load input data and iterators
        :return:
            data_set: data_set
        """
        # load data from input file
        data_set = DataSet(train_data_path=TRAIN_DATA_PATH, test_data_path=TEST_DATA_PATH,
                           valid_data_path=VALID_DATA_PATH, embedding_path=EMBEDDING_PATH)
        data_set.load_data()
        return data_set

    @staticmethod
    def init_model(data_set, use_pos):
        """
        init_model method is written for loading model and
        define loss function and optimizer
        """
        # create model

        model = ELMo(elmo_model_path=ELMO_MODEL_PATH, batch_size=BATCH_SIZE,
                     elmo_output_dim=EMBEDDING_DIM, lstm_hidden_dim=HIDDEN_DIM, lstm_layers=N_LAYERS,
                     bidirectional=BIDIRECTIONAL, output_size=OUTPUT_DIM,
                     pad_idx=data_set.pad_idx_dict["token_pad_idx"],
                     idx2word=data_set.text_field.vocab.itos,
                     start_dropout=START_DROPOUT, middle_dropout=MIDDLE_DROPOUT, final_dropout=FINAL_DROPOUT)

        # initializing model parameters
        model.apply(init_weights)
        logging.info("create model.")

        logging.info(f"The model has {count_parameters(model):,} trainable parameters")

        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # define loss function
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(data_set.class_weight))

        # load model into GPU
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        return model, criterion, optimizer

    @staticmethod
    def create_augmentation(data_set):
        """
        create_augmentation method is written for create augmentation class
        and define augmentation methods
        :param data_set: data_set class
        :return:
            augmentation_class: augmentation class
            augmentation_methods: augmentation method dictionary

        """
        word2idx = data_set.text_field.vocab.stoi
        idx2word = data_set.text_field.vocab.itos
        vocabs = list(word2idx.keys())
        augmentation_class = Augmentation(word2idx, idx2word, vocabs)

        # augmentation method dictionary
        augmentation_methods = {
            "delete_randomly": True,
            "replace_similar_words": True,
            "swap_token": True
        }
        return augmentation_class, augmentation_methods

    def eval_test_augmentation(self, best_test_f_score_model, data_set, aug_class):
        """
        eval_test_augmentation method is written for test augmentation
        :param best_val_loss_model: model path
        :param best_test_f_score_model: model path
        :param data_set: data_set class
        :param aug_class: augmentation class
        """
        log_file = open(TEST_AUG_LOG_PATH, "w")
        model, _, _ = self.init_model(data_set, USE_POS)

        model.load_state_dict(torch.load(MODEL_PATH + best_test_f_score_model, map_location=DEVICE))
        evaluate_parameters_dict = evaluate_aug_text(model=model, include_length=False,
                                                     iterator=data_set.iterator_dict["test_iterator"],
                                                     augmentation_class=aug_class)

        test_aug_result_log(log_file, evaluate_parameters_dict)

    def run(self, augmentation=USE_AUG):
        """
        run method is written for running model
        """
        data_set = self.load_data_set()
        model, criterion, optimizer = self.init_model(data_set, USE_POS)

        best_validation_loss = float("inf")
        best_test_f_score = 0.0

        losses_dict = dict()
        acc_dict = dict()
        losses_dict["train_loss"] = []
        losses_dict["dev_loss"] = []
        losses_dict["test_loss"] = []
        acc_dict["train_acc"] = []
        acc_dict["dev_acc"] = []
        acc_dict["test_acc"] = []

        if augmentation:
            augmentation_class, augmentation_methods = self.create_augmentation(data_set)

        if STEP_LR:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        # start training model
        for epoch in range(N_EPOCHS):
            start_time = time.time()

            # train model on train data
            if augmentation:
                train(model, data_set.iterator_dict["train_iterator"], optimizer, criterion,
                      augmentation_class, augmentation_methods)
            else:
                train(model, data_set.iterator_dict["train_iterator"], optimizer, criterion)

            if STEP_LR:
                scheduler.step()

            # compute model result on train data
            train_log_dict = evaluate(model, data_set.iterator_dict["train_iterator"], criterion)
            losses_dict["train_loss"].append(train_log_dict["loss"])
            acc_dict["train_acc"].append(train_log_dict["acc"])

            # compute model result on dev data
            dev_log_dict = evaluate(model, data_set.iterator_dict["dev_iterator"], criterion)
            losses_dict["dev_loss"].append(dev_log_dict["loss"])
            acc_dict["dev_acc"].append(dev_log_dict["acc"])

            # compute model result on test data
            test_log_dict = evaluate(model, data_set.iterator_dict["test_iterator"], criterion)
            losses_dict["test_loss"].append(test_log_dict["loss"])
            acc_dict["test_acc"].append(test_log_dict["acc"])

            end_time = time.time()

            # calculate epoch time
            epoch_mins, epoch_secs = process_time(start_time, end_time)

            # save model when fscore of test data is increase
            if test_log_dict["total_f_score"] > best_test_f_score:
                best_test_f_score = test_log_dict["total_f_score"]
                torch.save(model.state_dict(), MODEL_PATH + "best_model.pt")

            # show model result
            logging.info(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            model_result_log(train_log_dict, dev_log_dict, test_log_dict)

            # save model result in log file
            self.log_file.write(f"Epoch: {epoch + 1:02} | Epoch Time: "
                                f"{epoch_mins}m {epoch_secs}s\n")
            model_result_save(self.log_file, train_log_dict, dev_log_dict, test_log_dict)

        # save final model
        torch.save(model.state_dict(), MODEL_PATH + "final_model.pt")

        # plot curve
        draw_curves(train_acc=acc_dict["train_acc"], validation_acc=acc_dict["dev_acc"],
                         test_acc=acc_dict["test_acc"], train_loss=losses_dict["train_loss"],
                         validation_loss=losses_dict["dev_loss"],
                         test_loss=losses_dict["test_loss"])

        if TEST_USE_AUG and USE_AUG:
            self.eval_test_augmentation('best_model.pt', data_set, augmentation_class)
        elif TEST_USE_AUG:
            augmentation_class, augmentation_methods = self.create_augmentation(data_set)
            self.eval_test_augmentation('best_model.pt', data_set, augmentation_class)


if __name__ == "__main__":
    MYCLASS = RunModel()
    MYCLASS.run()
