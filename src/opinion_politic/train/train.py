"""
train.py is written for train model
"""

import logging
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from opinion_politic.utils.evaluation_helper import categorical_accuracy
from opinion_politic.config.rcnn_config import IS_ELMO

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_augmentation(text, text_lengths, label, augmentation_class, augmentation_methods):
    """
    batch_augmentation method is written for augment input batch
    :param text: input text
    :param text_lengths: text length
    :param label: label
    :param augmentation_class:  augmentation class
    :param augmentation_methods: augmentation methods dictionary
    :return:
    """
    augmentation_text, augmentation_label = list(), list()
    for txt, length, lbl in zip(text.tolist(), text_lengths.tolist(), label.tolist()):
        augment_sen = augmentation_class.__run__(txt, length, augmentation_methods)
        for sen in augment_sen:
            augmentation_text.append(sen)
            augmentation_label.append(lbl)
    tensor_augmentation_text = torch.FloatTensor(augmentation_text).long()
    tensor_augmentation_label = torch.FloatTensor(augmentation_label).long()

    augmented_text, augmented_label = tensor_augmentation_text, tensor_augmentation_label
    return augmented_text, augmented_label


def train(model, iterator, optimizer, criterion,  augmentation_class=None,
          augmentation_methods=None):
    """
    train method is written for train model
    :param model: model to be train
    :param iterator: data iterator to be train
    :param optimizer: optimizer function
    :param criterion: criterion function
    :param tag_pad_idx:
    :return:
        final_loss: final loss for epoch
        final_acc: accuracy for epoch
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    n_batch = 0

    # start training model
    for batch in iterator:
        n_batch += 1
        optimizer.zero_grad()

        text, text_lengths = batch.text
        label = batch.label

        if augmentation_class is not None:
            text, label = batch_augmentation(text, text_lengths, label,
                                             augmentation_class, augmentation_methods)

        # predict output
        # batch.text = [batch size, sent len]
        if IS_ELMO:
            predictions = model(text, text_lengths)
        else:
            predictions = model(text)

        # calculate loss
        # loss = criterion(predictions, label.float())
        loss = criterion(predictions, label)

        # calculate accuracy
        acc = categorical_accuracy(predictions, label)
        # acc = binary_accuracy(predictions, label)

        # back-propagate loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        # if (n_batch % (len(iterator)//5)) == 0:
        #     logging.info(f"\t train on: {(n_batch / len(iterator)) * 100:.2f}% of samples")
        #     logging.info(f"\t accuracy : {(epoch_acc/n_batch) * 100 :.2f}%")
        #     logging.info(f"\t loss : {(epoch_loss/n_batch):.4f}")
        #     logging.info("________________________________________________\n")

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """
    evaluate method is written for for evaluate model
    :param model: model to be evaluate
    :param iterator: data iterator to be evaluate
    :param criterion: criterion function
    :param tag_pad_idx:
    :return:
        final_loss: final loss for epoch
        final_acc: accuracy for epoch
        total_predict: all sample's prediction
        total_label: all sample's label
    """

    # define evaluate_parameters_dict to save output result
    evaluate_parameters_dict = {"loss": 0, "acc": 0, "precision": 0,
                                "recall": 0, "f_score": 0, "total_f_score": 0}

    # put model in evaluate model
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # predict input data
            text, text_lengths = batch.text
            label = batch.label
            if IS_ELMO:
                predictions = model(text, text_lengths)
            else:
                predictions = model(text)

                # calculate loss
            # loss = criterion(predictions, label.float())
            loss = criterion(predictions, label)

            # calculate accuracy
            acc = categorical_accuracy(predictions, label)
            # acc = binary_accuracy(predictions, label)

            # calculate precision, recall and f_score
            precision, recall, fscore, _ = \
                precision_recall_fscore_support(y_true=label.cpu(),
                                                y_pred=np.argmax(predictions.cpu(), axis=1))
            # precision, recall, fscore, _ = \
            #     precision_recall_fscore_support(y_true=label,
            #                                     y_pred=torch.round(torch.sigmoid(predictions)))

            # calculate total f-score of all data
            total_f_score = f1_score(y_true=label.cpu(),
                                     y_pred=np.argmax(predictions.cpu(), axis=1),
                                     average="weighted")
            # total_f_score = f1_score(y_true=label,
            #                          y_pred=torch.round(torch.sigmoid(predictions)),
            #                          average="weighted")

            evaluate_parameters_dict["loss"] += loss.item()
            evaluate_parameters_dict["acc"] += acc.item()
            evaluate_parameters_dict["precision"] += precision
            evaluate_parameters_dict["recall"] += recall
            evaluate_parameters_dict["f_score"] += fscore
            evaluate_parameters_dict["total_f_score"] += total_f_score

    evaluate_parameters_dict["loss"] = evaluate_parameters_dict["loss"] / len(iterator)
    evaluate_parameters_dict["acc"] = evaluate_parameters_dict["acc"] / len(iterator)
    evaluate_parameters_dict["precision"] = evaluate_parameters_dict["precision"] / len(iterator)
    evaluate_parameters_dict["recall"] = evaluate_parameters_dict["recall"] / len(iterator)
    evaluate_parameters_dict["f_score"] = evaluate_parameters_dict["f_score"] / len(iterator)
    evaluate_parameters_dict["total_f_score"] = evaluate_parameters_dict["total_f_score"] / len(iterator)

    return evaluate_parameters_dict


def test_augmentation(text, text_lengths, augmentation_class):
    """
    test_augmentation method is written for augment input text in evaluation
    :param text: input text
    :param text_lengths: text length
    :param augmentation_class:  augmentation class
    :return:
    """
    augmentation_text = augmentation_class.test_augment(text, text_lengths)
    augmentation_text.append(text)
    augmentation_text = torch.FloatTensor(augmentation_text).long()
    return augmentation_text


def evaluate_aug_text(model, iterator, include_length=False, augmentation_class=None):
    """
    evaluate_aug_text method is written for text augmentation for test sample
    :param model: model to be evaluate
    :param iterator: data iterator to be evaluate
    :param include_length: if true input length is given to the model
    :param augmentation_class: augmentation class
    :return:
        acc: accuracy of all  data
        precision: precision for each class of data
        recall: recall for each class of data
        f-score: F1-score for each class of data
        total_fscore: F1-score of all  data
    """
    # define evaluate_parameters_dict to save output result
    evaluate_parameters_dict = {"acc": 0, "precision": 0, "recall": 0,
                                "f-score": 0, "total_fscore": 0}
    total_predict = []
    total_label = []

    def most_frequent(pred):
        occurence_count = Counter(pred)
        return occurence_count.most_common(1)[0][0]

    # put model in evaluate model
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            # predict input data
            text, text_lengths = batch.text
            label = batch.label

            for sample, length, lbl in zip(text.tolist(), text_lengths.tolist(), label.tolist()):
                augment_sample = test_augmentation(sample, length, augmentation_class).to(DEVICE)
                if include_length:
                    aug_pred = model(augment_sample, length).tolist()
                else:
                    aug_pred = model(augment_sample).tolist()
                res = [np.argmax(i) for i in aug_pred]
                pred = most_frequent(res)
                total_predict.append(pred)
                total_label.append(lbl)
                # predictions = torch.FloatTensor(predictions).to(DEVICE)

    evaluate_parameters_dict["acc"] = accuracy_score(y_true=total_label, y_pred=total_predict)

    # calculate precision, recall and f_score
    evaluate_parameters_dict["precision"], evaluate_parameters_dict["recall"],\
        evaluate_parameters_dict["f-score"], _ = \
        precision_recall_fscore_support(y_true=total_label,
                                        y_pred=total_predict)

    # calculate total f-score of all data
    evaluate_parameters_dict["total_fscore"] = f1_score(y_true=total_label,
                                                        y_pred=total_predict,
                                                        average="weighted")

    return evaluate_parameters_dict
