"""
apcnn_lstm_model.py is written for Attention-based parallel cnn lstm model
"""

import torch
from torch import nn
import torch.nn.functional as F


class APCNN_LSTM(nn.Module):
    """
    In this class we implement APCNN_LSTM model
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.fix_len = kwargs["fix_len"]
        self.lstm_input_shape = self.lstm_input(kwargs["filter_sizes"], kwargs['fix_len'])

        self.embedding = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                      embedding_dim=kwargs["embedding_dim"],
                                      padding_idx=kwargs["pad_idx"])
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs['n_filters'],
                      kernel_size=(fs, kwargs['embedding_dim']))
            for fs in kwargs['filter_sizes']])

        self.lstm_r = nn.LSTM(self.lstm_input_shape,
                              hidden_size=kwargs["hidden_dim"],
                              num_layers=kwargs["n_layers"],
                              bidirectional=kwargs["bidirectional"],
                              dropout=kwargs["middle_dropout"])

        self.lstm_l_1 = nn.LSTM(kwargs["embedding_dim"],
                                hidden_size=kwargs["hidden_dim"],
                                num_layers=kwargs["n_layers"],
                                bidirectional=kwargs["bidirectional"],
                                dropout=kwargs["middle_dropout"])

        self.lstm_l_2 = nn.LSTM(kwargs["embedding_dim"] + (2 * kwargs["hidden_dim"]),
                                hidden_size=kwargs["hidden_dim"],
                                num_layers=kwargs["n_layers"],
                                bidirectional=kwargs["bidirectional"],
                                dropout=kwargs["middle_dropout"])

        self.W_s1 = nn.Linear(2 * kwargs["hidden_dim"], 350)
        self.W_s2 = nn.Linear(350, 30)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(30 * 2 * kwargs["hidden_dim"], 2000),
            nn.Linear(2000, kwargs["output_size"])
        )

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

    @staticmethod
    def lstm_input(filter_sizes, fix_len):
        lstm_input_shape = 0
        for item in filter_sizes:
            lstm_input_shape += (fix_len - item + 1)
        return lstm_input_shape

    def attention_net(self, lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, text):
        # text.size() = [batch size, sent len]

        text_embedded = self.start_dropout(self.embedding(text))
        # embedded.size() = [batch size, sent len, emb dim]

        r_embedded = text_embedded.unsqueeze(1)
        r_conved = [F.relu(conv(r_embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        r_conved = torch.cat(r_conved, dim=2)
        r_conved = r_conved.permute(1, 0, 2)
        # conved = [n_filter, batch size, sum[sent len - filter_sizes[n] + 1]]
        r_lstm_output, (_, _) = self.lstm_r(r_conved)
        # output.size() = (n_filter, batch_size, 2*hidden_size)

        l_embedded = text_embedded.permute(1, 0, 2)
        l_lstm_output_1, (_, _) = self.lstm_l_1(l_embedded)
        residual_1 = torch.cat((l_lstm_output_1, l_embedded), dim=2)
        l_lstm_output_2, (_, _) = self.lstm_l_2(residual_1)
        # output.size() = [sent len, batch size, hid dim * n directions]

        lstm_concat = torch.cat((l_lstm_output_2, r_lstm_output), dim=0)
        lstm_concat = lstm_concat.permute(1, 0, 2)
        # output.size() = [batch size, fix_len + n_filters, hid dim * n directions]

        attn_weight_matrix = self.attention_net(lstm_concat)
        # attn_weight_matrix.size() = (batch_size, r, fix_len + n_filters)

        hidden_matrix = torch.bmm(attn_weight_matrix, lstm_concat)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)

        final_output = self.fully_connected_layers(hidden_matrix.view(-1,
                                                   hidden_matrix.size()[1] * hidden_matrix.size()[2]))

        return final_output


if __name__ == '__main__':
    model = APCNN_LSTM(vocab_size=2000, embedding_dim=300, hidden_dim=256, output_size=2,
                       n_layers=2, bidirectional=True, start_dropout=0.5, middle_dropout=0.2,
                       final_dropout=0.2, pad_idx=1, n_filters=120, filter_sizes=[3, 4, 5], fix_len=70)
    x = torch.rand((1, 70))
    model.forward(x.long())
