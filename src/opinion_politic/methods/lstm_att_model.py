"""
lstm_att_model.py is written for LSTM_ATT model
"""

import torch
from torch import nn
import torch.nn.functional as F


class LSTM_ATT(nn.Module):
    """
    In this class we implement LSTM model for sentiment
    """
    def __init__(self, **kwargs):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=kwargs["vocab_size"],
                                      embedding_dim=kwargs["embedding_dim"],
                                      padding_idx=kwargs["pad_idx"])

        self.lstm = nn.LSTM(kwargs["embedding_dim"],
                            hidden_size=kwargs["hidden_dim"],
                            num_layers=kwargs['n_layers'],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"])

        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2 * kwargs["hidden_dim"], 350)
        self.W_s2 = nn.Linear(350, 30)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(30 * 2 * kwargs["hidden_dim"], 2000),
            # nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(2000, kwargs["output_size"])
        )

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

    def attention_net(self, lstm_output):

        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, text):
        # text.size() = [batch size, sent len]

        # pass text through embedding layer
        # embedded.size() = [batch size, sent len, emb dim]
        embedded = self.start_dropout(self.embedding(text))
        embedded = embedded.permute(1, 0, 2)

        output, (h_n, c_n) = self.lstm(embedded)
        output = output.permute(1, 0, 2)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)

        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)

        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        # Let's now concatenate the hidden_matrix and connect it to the fully connected layer.

        final_output = self.fully_connected_layers(hidden_matrix.view(-1,
                                                   hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        # final_output.size() = (batch_size, output_size)
        return final_output


if __name__ == '__main__':
    model = LSTM_ATT(vocab_size=2000, embedding_dim=300, hidden_dim=256, output_size=2,
                     n_layers=2, bidirectional=True, start_dropout=0.5, middle_dropout=0.2,
                     pad_idx=1, final_dropout=0.2)
    x = torch.rand((64, 150))
    model.forward(x.long())

