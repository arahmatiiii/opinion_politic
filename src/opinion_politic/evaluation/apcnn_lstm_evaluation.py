"""
apcnn_lstm_evaluation.py is written for predict apcnn_lstm models
"""

import torch
import hazm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PREDICT_SAMPLE_APCNN_LSTM():
    '''
    class for evaluation cnn lstm model
    '''
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.tokenizer = hazm.word_tokenize
        self.text_field = kwargs['text_field']

    def predict_apcnn_lstm(self, sentence, min_len=70):
        self.model.eval()
        tokenized = self.tokenizer(sentence)
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [self.text_field.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        tensor = tensor.unsqueeze(0)
        preds = self.model(tensor)
        max_preds = preds.argmax(dim=1)
        return max_preds.item()
