"""
bert_cnn_model.py is written for bert cnn model
"""

import torch
from torch import nn
import torch.nn.functional as F


class BertCnn(nn.Module):
    """
    In this class we implement Bert Cnn model
    """
    def __init__(self, **kwargs):

        super().__init__()

        self.bert = kwargs['bert']

        embedding_dim = kwargs['bert'].config.to_dict()['hidden_size']

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs['n_filters'],
                      kernel_size=(fs, embedding_dim))
            for fs in kwargs['filter_sizes']])

        self.fc = nn.Linear(len(kwargs['filter_sizes']) * kwargs['n_filters'],
                            kwargs["output_dim"])

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

    def forward(self, text):

        # text.size() = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        # embedded.size() = [batch size, sent len, 768]

        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cnn_cat = self.middle_dropout(torch.cat(pooled, dim=1))
        # cat_cnn = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cnn_cat)


if __name__ == '__main__':

    model = BertCnn(bert=bert_model, n_filters=128, filter_sizes=[3, 4, 5],
                    bidirectional=True, start_dropout=0.2,
                    middle_dropout=0.2, final_dropout=0.2, output_dim=2)

    x = torch.rand((150, 64))
    model.forward(x.long())
