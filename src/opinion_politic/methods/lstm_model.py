"""
lstm_model.py is written for LSTM model
"""

import torch
from torch import nn


class LSTM(nn.Module):
    """
    In this class we implement LSTM model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                      embedding_dim=kwargs["embedding_dim"],
                                      padding_idx=kwargs["pad_idx"])

        self.lstm = nn.LSTM(kwargs["embedding_dim"],
                            hidden_size=kwargs["hidden_dim"],
                            num_layers=kwargs["n_layers"],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"] if kwargs["n_layers"] > 1 else 0)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=kwargs["hidden_dim"] * 2 if kwargs["bidirectional"] else
                      kwargs["hidden_dim"],
                      out_features=512),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=256, out_features=kwargs["output_size"]))

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])

    def forward(self, text):

        # text.size() = [batch size, sent len]

        # pass text through embedding layer
        # embedded.size() = [batch size, sent len, emb dim]
        embedded = self.start_dropout(self.embedding(text))

        embedded = embedded.permute(1, 0, 2)

        # pass embeddings into LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        # output.size() = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        if self.lstm.bidirectional:
            hidden_can = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden_can = self.middle_dropout(nn.ReLU()(hidden_can))
        else:
            hidden_can = hidden[-1, :, :]
            hidden_can = self.middle_dropout(nn.ReLU()(hidden_can))

        return self.fully_connected_layers(hidden_can)


# if __name__ == '__main__':
#     model = LSTM(vocab_size=2000, embedding_dim=300, hidden_dim=128, output_size=2, n_layers=3,
#                  bidirectional=True, start_dropout=0.5, middle_dropout=0.2,
#                  pad_idx=1, final_dropout=0.2)
#     x = torch.rand((150, 64))
#     model.forward(x.long())
