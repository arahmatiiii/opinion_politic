"""
abcdn_model.py is written for Cnn ABCDM model
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ABCDM(nn.Module):
    """
    In this class we implement ABCDM model
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
            nn.Conv1d(in_channels=1,
                      out_channels=kwargs['n_filters'],
                      kernel_size=fs)
            for fs in kwargs['filter_sizes']])

        self.lstm = nn.LSTM(kwargs["embedding_dim"], kwargs["hidden_dim"] // 2,
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"] if kwargs["n_layers"] > 1 else 0)

        self.gru = nn.GRU(kwargs["embedding_dim"], kwargs["hidden_dim"] // 2,
                          bidirectional=kwargs["bidirectional"],
                          dropout=kwargs["middle_dropout"] if kwargs["n_layers"] > 1 else 0)

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=8 * kwargs['n_filters'], out_features=64),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=64, out_features=kwargs["output_size"]),
            # nn.Sigmoid(),
        )

        self.batchnorm = nn.BatchNorm1d(num_features=8 * kwargs['n_filters'])
        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])

    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2)).unsqueeze(2)
        return torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)

    def forward(self, text):
        # text.size() = [batch size, sent len]

        embedded = self.start_dropout(self.embedding(text))
        embedded = embedded.permute(1, 0, 2)

        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(embedded)
        gru_output, gru_hidden = self.gru(embedded)

        lstm_attn_output = self.attention(lstm_output, lstm_hidden)
        gru_attn_output = self.attention(gru_output, gru_hidden)

        lstm_attn_output = lstm_attn_output.unsqueeze(1)
        gru_attn_output = gru_attn_output.unsqueeze(1)

        lstm_conved = [F.relu(conv(lstm_attn_output)) for conv in self.convs]
        gru_conved = [F.relu(conv(gru_attn_output)) for conv in self.convs]

        lstm_max_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in lstm_conved]
        lstm_max_pooled = self.start_dropout(nn.ReLU()(torch.cat(lstm_max_pooled, dim=1)))

        gru_max_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in gru_conved]
        gru_max_pooled = self.start_dropout(nn.ReLU()(torch.cat(gru_max_pooled, dim=1)))

        lstm_avg_pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in lstm_conved]
        lstm_avg_pooled = self.start_dropout(nn.ReLU()(torch.cat(lstm_avg_pooled, dim=1)))

        gru_avg_pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in gru_conved]
        gru_avg_pooled = self.start_dropout(nn.ReLU()(torch.cat(gru_avg_pooled, dim=1)))

        poled_con = torch.cat((lstm_max_pooled, lstm_avg_pooled, gru_max_pooled, gru_avg_pooled), dim=1)

        normed = self.batchnorm(self.start_dropout(nn.ReLU()(poled_con)))

        final_output = self.fully_connected_layers(self.start_dropout(nn.ReLU()((normed))))
        return final_output


if __name__ == '__main__':
    object = ABCDM(vocab_size=2000, embedding_dim=300, hidden_dim=128,
                   n_layers=1, pad_idx=0, bidirectional=True,
                   start_dropout=0.35, middle_dropout=0.35,
                   final_dropout=0.35, output_size=2, filter_sizes=[5, 7], n_filters=100)

    x = torch.rand((64, 150))
    object.forward(x.long())
