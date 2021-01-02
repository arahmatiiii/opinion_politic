"""
config.py is written for LSTM model
"""

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 3
EMBEDDING_DIM = 300
START_DROPOUT = 0.5
MIDDLE_DROPOUT = 0.5
FINAL_DROPOUT = 0.5
BATCH_SIZE = 64


HIDDEN_DIM = 50
N_LAYERS = 1
BIDIRECTIONAL = True



USE_AUG = False
USE_POS = False
POS_ONE_HOT = True

TRAIN_DATA_PATH = "../data/Processed/train_data_normed.csv"
TEST_DATA_PATH = "../data/Processed/test_data_normed.csv"
VALID_DATA_PATH = "../data/Processed/valid_data_normed.csv"
EMBEDDING_PATH = "../data/Embeddings/wor2vec_skipgram300d.txt"

EVAL_USER_PATH = "../data/Intermadiate/evaluation_user.csv"
EVAL_USER_DATA_PATH = "../data/Intermadiate/evaluation_user_data/"


palce_save = 'Lstm/ID_0'
TEXT_FIELD_PATH = "../models/"+palce_save+"/Fields/text_field.Field"
LABEL_FIELD_PATH = "../models/"+palce_save+"/Fields/label_field.Field"
POS_FIELD_PATH = "../models/"+palce_save+"/Fields/pos_field.Field"
MODEL_PATH = "../models/"+palce_save+"/"

detiels = 'test'
LOG_PATH = "../models/"+palce_save+"/Logs/log "+detiels+".txt"
LOSS_CURVE_PATH = "../models/"+palce_save+"/Curves/loss_curve "+detiels+".png"
ACC_CURVE_PATH = "../models/"+palce_save+"/Curves/accuracy_curve "+detiels+".png"
