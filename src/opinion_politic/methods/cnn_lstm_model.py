"""
Cnn_Lstm_model.py is written for Cnn Lstm model
"""

import torch
from torch import nn
import torch.nn.functional as F


class CNNLSTM(nn.Module):
    """
    In this class we implement Cnn LSTM model
    """

    def __init__(self, **kwargs):
        '''
        text_vocab_size, lemma_vocab_size, pos_vocab_size,
                 text_embedding_dim, lemma_embedding_dim, pos_embedding_dim,
                 text_pad_idx, lemma_pad_idx, pos_pad_idx,
                 filter_sizes, n_filters, dropout, output_dim
        :param kwargs:
        '''
        super().__init__()

        self.embedding = nn.Embedding(kwargs['vocab_size'], kwargs['embedding_dim'],
                                      padding_idx=kwargs['pad_idx'])

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs['n_filters'],
                      kernel_size=(fs, kwargs['embedding_dim']))
            for fs in kwargs['filter_sizes']])

        self.lstm = nn.LSTM(kwargs["embedding_dim"],
                            hidden_size=kwargs["hidden_dim"],
                            num_layers=kwargs["n_layers"],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"] if kwargs["n_layers"] > 1 else 0)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=(len(kwargs['filter_sizes']) * kwargs['n_filters']) + (kwargs["hidden_dim"] * 2 if kwargs["bidirectional"] else
                      kwargs["hidden_dim"]),
                      out_features=256),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=128, out_features=kwargs["output_size"]))

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])



    def forward(self, text):
        # text.size() = [batch size, sent len]

        # pass text through embedding layer
        # embedded.size() = [batch size, sent len, emb dim]

        embedded = self.start_dropout(self.embedding(text))

        cnn_embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(cnn_embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cnn_cat = self.middle_dropout(nn.ReLU()(torch.cat(pooled, dim=1)))

        lstm_embedded = embedded.permute(1, 0, 2)
        outputs, (hidden, cell) = self.lstm(lstm_embedded)
        if self.lstm.bidirectional:
            hidden_can = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
            hidden_can = self.middle_dropout(nn.ReLU()(hidden_can))
        else:
            hidden_can = hidden[-1, :, :]
            hidden_can = self.middle_dropout(nn.ReLU()(hidden_can))

        cnn_lstm_can = torch.cat((cnn_cat, hidden_can), dim=1)

        final_output = self.fully_connected_layers(cnn_lstm_can)
        return final_output



# if __name__ == '__main__':
#     model = CNNLSTM(vocab_size=2000,
#                     embedding_dim=300,
#                     pad_idx=0,
#                     n_filters=128, filter_sizes=[3, 4, 5],
#                     hidden_dim=128, n_layers=2, bidirectional=True,
#                     start_dropout=0.35, middle_dropout=0.35, final_dropout=0.35,
#                     output_size=2)
#     x = torch.rand((150, 64))
#     model.forward(x.long())
