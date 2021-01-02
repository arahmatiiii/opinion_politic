"""
cnn_evaluation.py is written for predict cnn models
"""

import torch
import hazm
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PREDICT_SAMPLE_CNN():
    '''
    class for evaluation cnn model
    '''
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.tokenizer = hazm.word_tokenize
        self.text_field = kwargs['text_field']

    def predict_cnn(self, sentence, min_len=5):
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
