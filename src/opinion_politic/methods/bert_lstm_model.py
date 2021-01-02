"""
parstbert_model.py is written for pars bert model
"""

import torch
from torch import nn
from opinion_politic.config.parsbert_config import FIX_LEN


class ParsBert(nn.Module):
    """
    In this class we implement Pars Bert model
    """
    def __init__(self, **kwargs):

        super().__init__()

        self.bert = kwargs['bert']
        self.output_dim = kwargs["output_dim"]

        embedding_dim = kwargs['bert'].config.to_dict()['hidden_size']

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(FIX_LEN * 768, out_features=512),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=256, out_features=kwargs["output_dim"]))

        self.fc = nn.Linear(in_features=FIX_LEN*768, out_features=kwargs['output_dim'])
        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])

    def forward(self, text):

        # text.size() = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded.size() = [batch size, sent len, 768]

        embedded = torch.flatten(embedded, start_dim=1)
        # embedded.size() = [batch size, sent len * 768]

        predictions = self.fc(embedded)

        return predictions


# if __name__ == '__main__':
#     model = ParsBert(bert=bert_model, start_dropout=0.35,
#                      middle_dropout=0.35, final_dropout=0.35, output_dim=2)
#     x = torch.rand((150, 64))
#     model.forward(x.long())
