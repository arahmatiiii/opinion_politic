"""
config.py is written for transformer_model
"""

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EPOCHS = 3
EMBEDDING_DIM = 300
BATCH_SIZE = 64
OUTPUT_DIM = 2
HID_DIM = 256
ENC_LAYERS = 3
ENC_HEADS = 8
ENC_PF_DIM = 512
ENC_DROPOUT = 0.2
MAX_LEN = 70


IS_TRANSFORMER = True
STEP_LR = False
TEST_USE_AUG = False
USE_AUG = False
USE_POS = False
USE_STOPWORD = False

TRAIN_DATA_PATH = "../data/Processed/train_data_normed.csv"
TEST_DATA_PATH = "../data/Processed/test_data_normed.csv"
VALID_DATA_PATH = "../data/Processed/valid_data_normed.csv"
EMBEDDING_PATH = "../data/Embeddings/wor2vec_skipgram300d.txt"
STOPWORDS_PATH = "../data/Processed/stopwords.xlsx"

EVAL_USER_PATH = "../data/Intermadiate/evaluation_user.csv"
EVAL_USER_DATA_PATH = "../data/Intermadiate/evaluation_user_data/"

palce_save = 'Transformer/ID_0'
TEXT_FIELD_PATH = "../models/"+palce_save+"/Fields/text_field.Field"
LABEL_FIELD_PATH = "../models/"+palce_save+"/Fields/label_field.Field"
POS_FIELD_PATH = "../models/"+palce_save+"/Fields/pos_field.Field"
MODEL_PATH = "../models/"+palce_save+"/"

detiels = 'test'
LOG_PATH = "../models/"+palce_save+"/Logs/log "+detiels+".txt"
TEST_AUG_LOG_PATH = "../models/"+palce_save+"/Logs/log_aug "+detiels+".txt"
LOSS_CURVE_PATH = "../models/"+palce_save+"/Curves/loss_curve "+detiels+".png"
ACC_CURVE_PATH = "../models/"+palce_save+"/Curves/accuracy_curve "+detiels+".png"
