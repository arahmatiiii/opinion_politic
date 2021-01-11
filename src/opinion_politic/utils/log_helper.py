"""
log_helper.py is a file to write methods which use for better log
"""

import logging
import matplotlib.pyplot as plt
from opinion_politic.config.dpcnn_config import LOSS_CURVE_PATH, ACC_CURVE_PATH


logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def count_parameters(input_model):
    """
    count_parameters method is written for calculate number of model's parameter
    :param input_model: model
    :return:
        num_parameters: number of model parameters
    """
    num_parameters = sum(p.numel() for p in input_model.parameters() if p.requires_grad)
    return num_parameters


def process_time(s_time, e_time):
    """
    process_time method is written for calculate time
    :param s_time: start time
    :param e_time: end time
    :return:
        elapsed_mins: Minutes of process
        elapsed_secs: Seconds of process
    """
    elapsed_time = e_time - s_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def model_result_log(train_log_dict, dev_log_dict, test_log_dict):
    """
    model_result_log method is writen for show model result on each epoch
    :param train_log_dict: dictionary of train data result
    :param dev_log_dict: dictionary of validation data result
    :param test_log_dict: dictionary of test data result
    """
    logging.info(f"\tTrain. Loss: {train_log_dict['loss']:.4f} | "
                 f"Train. Acc: {train_log_dict['acc'] * 100:.2f}%")
    logging.info(f"\t Val. Loss: {dev_log_dict['loss']:.4f} |  "
                 f"Val. Acc: {dev_log_dict['acc'] * 100:.2f}%")
    logging.info(f"\t Test. Loss: {test_log_dict['loss']:.4f} |  "
                 f"Test. Acc: {test_log_dict['acc'] * 100:.2f}%")

    logging.info(f"\t Train. Precision: {train_log_dict['precision']}")
    logging.info(f"\t Train. Recall: {train_log_dict['recall']}")
    logging.info(f"\t Train. F1_Score: {train_log_dict['f_score']}")
    logging.info(f"\t Train. Total F1 score: {train_log_dict['total_f_score']}")

    logging.info(f"\t Val. Precision: {dev_log_dict['precision']}")
    logging.info(f"\t Val. Recall: {dev_log_dict['recall']}")
    logging.info(f"\t Val. F1_Score: {dev_log_dict['f_score']}")
    logging.info(f"\t Val. Total F1 score: {dev_log_dict['total_f_score']}")

    logging.info(f"\t Test. Precision: {test_log_dict['precision']}")
    logging.info(f"\t Test. Recall: {test_log_dict['recall']}")
    logging.info(f"\t Test. F1_Score: {test_log_dict['f_score']}")
    logging.info(f"\t Test. Total F1 score: {test_log_dict['total_f_score']}")

    logging.info("\n")


def model_result_save(log_file, train_log_dict, dev_log_dict, test_log_dict):
    """
    model_result_save method is writen for save model result on each epoch
    :param log_file: log text file
    :param train_log_dict: dictionary of train data result
    :param dev_log_dict: dictionary of validation data result
    :param test_log_dict: dictionary of test data result
    """
    log_file.write(f"\tTrain Loss: {train_log_dict['loss']:.4f} | "
                   f"Train Acc: {train_log_dict['acc'] * 100:.2f}%\n")
    log_file.write(f"\t Val. Loss: {dev_log_dict['loss']:.4f} |  "
                   f"Val. Acc: {dev_log_dict['acc'] * 100:.2f}%\n")
    log_file.write(f"\t Test. Loss: {test_log_dict['loss']:.4f} |  "
                   f"Test. Acc: {test_log_dict['acc'] * 100:.2f}%\n")

    log_file.write(f"\t Train. Precision: {train_log_dict['precision']}\n")
    log_file.write(f"\t Train. Recall: {train_log_dict['recall']}\n")
    log_file.write(f"\t Train. F1_Score: {train_log_dict['f_score']}\n")
    log_file.write(f"\t Train. Total F1 score: {train_log_dict['total_f_score']}\n")

    log_file.write(f"\t Val. Precision: {dev_log_dict['precision']}\n")
    log_file.write(f"\t Val. Recall: {dev_log_dict['recall']}\n")
    log_file.write(f"\t Val. F1_Score: {dev_log_dict['f_score']}\n")
    log_file.write(f"\t Val. Total F1 score: {dev_log_dict['total_f_score']}\n")

    log_file.write(f"\t Test. Precision: {test_log_dict['precision']}\n")
    log_file.write(f"\t Test. Recall: {test_log_dict['recall']}\n")
    log_file.write(f"\t Test. F1_Score: {test_log_dict['f_score']}\n")
    log_file.write(f"\t Test. Total F1 score: {test_log_dict['total_f_score']}\n")

    log_file.write("\n")
    log_file.flush()


def test_aug_result_log(log_file, evaluate_parameters_dict):
    logging.info(f"\t Test. acc: {evaluate_parameters_dict['acc']}\n")
    logging.info(f"\t Test. Precision: {evaluate_parameters_dict['precision']}\n")
    logging.info(f"\t Test. Recall: {evaluate_parameters_dict['recall']}\n")
    logging.info(f"\t Test. F1_Score: {evaluate_parameters_dict['f-score']}\n")
    logging.info(f"\t Test. Total F1 score: {evaluate_parameters_dict['total_fscore']}\n")

    log_file.write(f"\t Test. acc: {evaluate_parameters_dict['acc']}\n")
    log_file.write(f"\t Test. Precision: {evaluate_parameters_dict['precision']}\n")
    log_file.write(f"\t Test. Recall: {evaluate_parameters_dict['recall']}\n")
    log_file.write(f"\t Test. F1_Score: {evaluate_parameters_dict['f-score']}\n")
    log_file.write(f"\t Test. Total F1 score: {evaluate_parameters_dict['total_fscore']}\n")

    log_file.write("\n")
    log_file.flush()


def draw_curves(**kwargs):
    """
    draw_curves method is written for drawing loss and accuracy curve
    """
    # plot loss curves
    plt.plot(kwargs["train_loss"], "r", label="train_loss")
    plt.plot(kwargs["validation_loss"], "b", label="validation_loss")
    plt.plot(kwargs["test_loss"], "g", label="test_loss")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("loss value")
    plt.savefig(LOSS_CURVE_PATH)

    # clear figure command
    plt.clf()

    # plot accuracy curves
    plt.plot(kwargs["train_acc"], "r", label="train_acc")
    plt.plot(kwargs["validation_acc"], "b", label="validation_acc")
    plt.plot(kwargs["test_acc"], "g", label="test_acc")
    plt.legend()
    plt.xlabel("number of epochs")
    plt.ylabel("accuracy value")
    plt.savefig(ACC_CURVE_PATH)