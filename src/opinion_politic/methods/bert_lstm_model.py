"""
bert_lstm_model.py is written for bert lstm model
"""

import torch
from torch import nn


class BertLstm(nn.Module):
    """
    In this class we implement Bert lstm model
    """
    def __init__(self, **kwargs):

        super().__init__()

        self.bert = kwargs['bert']

        embedding_dim = kwargs['bert'].config.to_dict()['hidden_size']

        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size=kwargs["hidden_dim"],
                            num_layers=kwargs["n_layers"],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"] if kwargs["n_layers"] > 1 else 0)

        self.fc = nn.Linear(kwargs["hidden_dim"] * 2 if kwargs["bidirectional"] else kwargs["hidden_dim"],
                            kwargs["output_dim"])

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

    def forward(self, text):

        # text.size() = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded.size() = [batch size, sent len, 768]

        embedded = embedded.permute(1, 0, 2)

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # output.size() = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        if self.lstm.bidirectional:
            hidden_can = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden_can = self.middle_dropout(hidden_can)
        else:
            hidden_can = hidden[-1, :, :]
            hidden_can = self.middle_dropout(hidden_can)

        return self.fc(hidden_can)


if __name__ == '__main__':
    model = BertLstm(bert=bert_model, hidden_dim=256, n_layers=1, bidirectional=True, start_dropout=0.2,
                     middle_dropout=0.2, final_dropout=0.2, output_dim=2)
    x = torch.rand((150, 64))
    model.forward(x.long())
