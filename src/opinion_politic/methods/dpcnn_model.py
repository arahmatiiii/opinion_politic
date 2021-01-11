"""
dpcnn_model.py is written for DpCnn model
"""

import torch
from torch import nn
import torch.nn.functional as F


class DPCNN(nn.Module):
    """
    In this class we implement DPCNN model
    """
    def __init__(self, **kwargs):
        super(DPCNN, self).__init__()
        self.embedding_dim = kwargs['embedding_dim']
        self.n_filters = kwargs['n_filters']
        self.output_size = kwargs["output_size"]

        self.embedding = nn.Embedding(kwargs['vocab_size'], kwargs['embedding_dim'],
                                      padding_idx=kwargs['pad_idx'])

        self.conv_region = nn.Conv2d(1, self.n_filters, (3, self.embedding_dim), stride=1)
        self.conv = nn.Conv2d(self.n_filters, self.n_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))
        self.relu = nn.ReLU()
        self.fully_connected_layer = nn.Linear(self.n_filters * 2, self.output_size)

        self.start_dropout = nn.Dropout(kwargs["start_dropout"])
        self.middle_dropout = nn.Dropout(kwargs["middle_dropout"])
        self.final_dropout = nn.Dropout(kwargs["final_dropout"])

    def forward(self, text):
        embedded = self.start_dropout(self.embedding(text)).unsqueeze(1)
        # [batch_size, 1, seq_len, emb_dim]

        result = self.conv_region(embedded)
        # [batch_size, num_filters, seq_len-3+1, 1]
        result = self.relu(self.padding1(result))
        # [batch_size, num_filters, seq_len, 1]

        result = self.conv(result)
        # [batch_size, num_filters, seq_len-3+1, 1]
        result = self.relu(self.padding1(result))
        # [batch_size, num_filters, seq_len, 1]

        result = self.conv(result)
        # [batch_size, num_filters, seq_len-3+1, 1]
        while result.size()[2] > 2:
            result = self._block(result)

        result = result.squeeze().flatten(1, 2)
        # [batch_size, num_filters * 2]
        final_output = self.fully_connected_layer(result)
        # [batch_size, output_dim]

        return final_output

    def _block(self, input_text):
        input_text = self.padding2(input_text)
        pooled_input_text = self.max_pool(input_text)

        input_text = F.relu(self.padding1(pooled_input_text))
        input_text = self.conv(input_text)

        input_text = F.relu(self.padding1(input_text))
        input_text = self.conv(input_text)

        # Short Cut
        out_text = input_text + pooled_input_text
        return out_text


if __name__ == '__main__':
    model = DPCNN(vocab_size=2000, embedding_dim=300, n_filters=256, output_size=2,
                  pad_idx=1, start_dropout=0.5, middle_dropout=0.2, final_dropout=0.2)
    x = torch.rand((64, 150))
    model.forward(x.long())
