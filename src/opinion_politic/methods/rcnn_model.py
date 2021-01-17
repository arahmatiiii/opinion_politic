"""
rcnn_model.py is written for Rcnn model
"""

import torch
from torch import nn
import torch.nn.functional as F


class RCNN(nn.Module):
    """
    In this class we implement RCNN model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                       embedding_dim=kwargs["embedding_dim"],
                                       padding_idx=kwargs["pad_idx"])

        self.lstm = nn.LSTM(kwargs["embedding_dim"],
                            hidden_size=kwargs["hidden_dim"],
                            num_layers=kwargs["n_layers"],
                            dropout=kwargs["middle_dropout"],
                            bidirectional=True)

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

        self.linear = nn.Linear(
            in_features=kwargs["embedding_dim"] + (2 * kwargs["hidden_dim"]),
            out_features=kwargs["hidden_dim"]
        )
        self.fully_connected_layer = nn.Linear(in_features=kwargs["hidden_dim"],
                                               out_features=kwargs["output_size"])

    def forward(self, input_batch):
        # input_batch.size() = [batch_size, sent_len]

        embedded = self.start_dropout(self.embedding(input_batch))
        # embedded.size() = [batch_size, sent_len, embedding_dim]

        embedded = embedded.permute(1, 0, 2)
        # embedded.size() = [sent_len, batch_size, embedding_dim]

        lstm_output, (_, _) = self.lstm(embedded)
        # output_1.size() = [sent_len, batch_size, hid_dim * num_directions]
        # hidden.size() = [num_layers * num_directions, batch_size, hid_dim]
        # cell.size() = [num_layers * num_directions, batch_size, hid_dim]

        input_features = torch.cat((lstm_output, embedded), 2).permute(1, 0, 2)
        # final_features.size() = [batch_size, sent_len, embedding_dim+2*hid_dim]
        linear_output = self.linear(input_features)
        # linear_output.size() = [batch_size, sent_len, hid_dim]

        linear_output = linear_output.permute(0, 2, 1)
        # linear_output.size() = [batch_size, hid_dim, sent_len]

        max_out_features = F.max_pool1d(linear_output, linear_output.size()[2]).squeeze(2)
        # max_out_features.size() = [batch_size, hid_dim]
        final_output = self.fully_connected_layer(max_out_features)

        return final_output


if __name__ == '__main__':
    model = RCNN(vocab_size=2000, embedding_dim=300, hidden_dim=256, output_size=2,
                 n_layers=2, bidirectional=True, start_dropout=0.5, middle_dropout=0.2,
                 pad_idx=1, final_dropout=0.2)
    x = torch.rand((64, 150))
    model.forward(x.long())