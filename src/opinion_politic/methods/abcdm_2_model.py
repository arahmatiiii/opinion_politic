"""
abcdn_2_model.py is written for Cnn ABCDM model
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ABCDM_2(nn.Module):
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

        self.lstm = nn.LSTM(kwargs["embedding_dim"],
                            hidden_size=kwargs["hidden_dim"],
                            num_layers=kwargs['n_layers'],
                            bidirectional=kwargs["bidirectional"],
                            dropout=kwargs["middle_dropout"])

        self.gru = nn.GRU(kwargs["embedding_dim"],
                          hidden_size=kwargs["hidden_dim"],
                          num_layers=kwargs['n_layers'],
                          bidirectional=kwargs["bidirectional"],
                          dropout=kwargs["middle_dropout"])

        self.W_s1 = nn.Linear(2 * kwargs["hidden_dim"], 350)
        self.W_s2 = nn.Linear(350, 30)

        self.fully_connected_layers = nn.Linear(in_features=8 * kwargs['n_filters'],
                                                out_features=kwargs["output_size"])

        self.batchnorm = nn.BatchNorm1d(num_features=8 * kwargs['n_filters'])

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

        embedded = self.start_dropout(self.embedding(text))
        embedded = embedded.permute(1, 0, 2)

        lstm_output, (_, _) = self.lstm(embedded)
        lstm_output = lstm_output.permute(1, 0, 2)

        gru_output, _ = self.gru(embedded)
        gru_output = gru_output.permute(1, 0, 2)

        lstm_attn_weight_matrix = self.attention_net(lstm_output)
        lstm_hidden_matrix = torch.bmm(lstm_attn_weight_matrix, lstm_output)

        gru_attn_weight_matrix = self.attention_net(gru_output)
        gru_hidden_matrix = torch.bmm(gru_attn_weight_matrix, gru_output)

        lstm_hidden_matrix = lstm_hidden_matrix.flatten(1, 2).unsqueeze(1)
        gru_hidden_matrix = gru_hidden_matrix.flatten(1, 2).unsqueeze(1)

        lstm_conved = [F.relu(conv(lstm_hidden_matrix)) for conv in self.convs]
        gru_conved = [F.relu(conv(gru_hidden_matrix)) for conv in self.convs]

        lstm_max_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in lstm_conved]
        lstm_max_pooled = self.middle_dropout(nn.ReLU()(torch.cat(lstm_max_pooled, dim=1)))

        gru_max_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in gru_conved]
        gru_max_pooled = self.middle_dropout(nn.ReLU()(torch.cat(gru_max_pooled, dim=1)))

        lstm_avg_pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in lstm_conved]
        lstm_avg_pooled = self.middle_dropout(nn.ReLU()(torch.cat(lstm_avg_pooled, dim=1)))

        gru_avg_pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in gru_conved]
        gru_avg_pooled = self.middle_dropout(nn.ReLU()(torch.cat(gru_avg_pooled, dim=1)))

        poled_con = torch.cat((lstm_max_pooled, lstm_avg_pooled, gru_max_pooled, gru_avg_pooled), dim=1)

        normed = self.batchnorm(poled_con)

        final_output = self.fully_connected_layers(normed)

        return final_output


if __name__ == '__main__':
    object = ABCDM_2(vocab_size=2000, embedding_dim=300, hidden_dim=128,
                        n_layers=1, pad_idx=0, bidirectional=True,
                        start_dropout=0.35, middle_dropout=0.35,
                        final_dropout=0.35, output_size=2, filter_sizes=[5, 7], n_filters=100)

    x = torch.rand((64, 150))
    object.forward(x.long())
