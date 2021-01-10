"""
dc_lstm_model.py is written for DC_LSTM model
"""

import torch
from torch import nn
import torch.nn.functional as F


class DC_LSTM(nn.Module):
    """
    In this class we implement DC_LSTM model for sentiment
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                      embedding_dim=kwargs["embedding_dim"],
                                      padding_idx=kwargs["pad_idx"])

        self.lstm_1 = nn.LSTM(kwargs["embedding_dim"],
                              hidden_size=kwargs["first_hidden_dim"],
                              num_layers=kwargs["n_layers"],
                              bidirectional=kwargs["bidirectional"],
                              dropout=kwargs["middle_dropout"])

        self.lstm_2 = nn.LSTM(kwargs["embedding_dim"] + (2 * kwargs["first_hidden_dim"]),
                              hidden_size=kwargs["second_hidden_dim"],
                              num_layers=kwargs["n_layers"],
                              bidirectional=kwargs["bidirectional"],
                              dropout=kwargs["middle_dropout"])

        self.lstm_3 = nn.LSTM(kwargs["embedding_dim"] + (2 * kwargs["first_hidden_dim"]) +
                              (2 * kwargs["second_hidden_dim"]),
                              hidden_size=kwargs["third_hidden_dim"],
                              num_layers=kwargs["n_layers"],
                              bidirectional=kwargs["bidirectional"],
                              dropout=kwargs["middle_dropout"])

        self.fully_connected_layers = nn.Linear(in_features=kwargs["second_hidden_dim"] * 2
                                        if kwargs["bidirectional"] else kwargs["hidden_dim"],
                                        out_features=kwargs["output_size"])

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

    def forward(self, text):
        # text.size() = [batch size, sent len]

        # pass text through embedding layer
        # embedded.size() = [batch size, sent len, emb dim]
        embedded = self.start_dropout(self.embedding(text))
        embedded = embedded.permute(1, 0, 2)

        # pass embeddings into LSTM
        output_1, (_, _) = self.lstm_1(embedded)
        # output.size() = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]

        residual_1 = torch.cat((output_1, embedded), dim=2)

        output_2, (_, _) = self.lstm_2(residual_1)

        residual_2 = torch.cat((output_1, output_2, embedded), dim=2)

        output_3, (_, _) = self.lstm_3(residual_2)
        output_3 = output_3.permute(1, 2, 0)
        # output_3.size() = [batch_size, hid_dim * num_directions, sent_len]

        avg_pooling = F.avg_pool1d(output_3, output_3.shape[2]).squeeze(2)
        # avg_pooling.size() = [batch_size, hid_dim * num_directions]

        avg_pooling = self.middle_dropout(avg_pooling)
        # avg_pooling.size() = [batch_size, sent_len]

        final_output = self.fully_connected_layers(avg_pooling)

        return final_output


if __name__ == '__main__':
    model = DC_LSTM(vocab_size=2000, embedding_dim=300, first_hidden_dim=128,
                    second_hidden_dim=128, third_hidden_dim=128,
                    output_size=2, n_layers=2,
                    bidirectional=True, start_dropout=0.2, middle_dropout=0.2,
                    final_dropout=0.2, pad_idx=1)

    x = torch.rand((64, 150))
    model.forward(x.long())

