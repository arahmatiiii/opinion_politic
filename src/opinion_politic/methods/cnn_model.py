"""
Cnn_model.py is written for Cnn model
"""

import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    In this class we implement Cnn model
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.embedding = nn.Embedding(kwargs['vocab_size'], kwargs['embedding_dim'],
                                      padding_idx=kwargs['pad_idx'])

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=kwargs['n_filters'],
                      kernel_size=(fs, kwargs['embedding_dim']))
            for fs in kwargs['filter_sizes']])

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(in_features=len(kwargs['filter_sizes']) * kwargs['n_filters'], out_features=512),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Dropout(kwargs["final_dropout"]),
            nn.Linear(in_features=256, out_features=kwargs['output_size']))

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])



    def forward(self, text):
        # text.size() = [batch size, sent len]

        # pass text through embedding layer
        # embedded.size() = [batch size, sent len, emb dim]
        embedded = self.start_dropout(self.embedding(text))

        embedded = embedded.unsqueeze(1)

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cnn_cat = self.middle_dropout(nn.ReLU()(torch.cat(pooled, dim=1)))
        # cat_cnn = [batch size, n_filters * len(filter_sizes)]

        return self.fully_connected_layers(cnn_cat)


# if __name__ == '__main__':
#     model = CNN(vocab_size=2000, embedding_dim=300, pad_idx=0, n_filters=128, filter_sizes=[3, 4, 5],
#                 start_dropout=0.1, middle_dropout=0.2, final_dropout=0.3, output_size=2)
#     x = torch.rand((150, 64))
#     model.forward(x.long())
