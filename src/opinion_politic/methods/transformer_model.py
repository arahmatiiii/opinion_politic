"""
transformer_model.py is written for transformer model
"""

import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                vocab_size, hid_dim, n_layers, n_heads, pf_dim,
                dropout, device, max_length=100):

        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hid_dim)
        self.tok_embedding.weight.requires_grad = True

        self.pos_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=hid_dim)
        self.pos_embedding.weight.requires_grad = True

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.max_length = max_length
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, text, text_mask):
        # text = [batch size, src len]
        # text_mask = [batch size, 1, 1, src len]

        batch_size = text.shape[0]
        text_len = text.shape[1]

        pos = torch.arange(0, text_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]

        text = self.dropout((self.tok_embedding(text) * self.scale) + self.pos_embedding(pos))
        # text = [batch size, src len, hid dim]

        for layer in self.layers:
            text = layer(text, text_mask)

        # src = [batch size, src len, hid dim]
        return text


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_mask):
        # text = [batch size, src len, hid dim]
        # text_mask = [batch size, 1, 1, src len]

        # self attention
        _text, _ = self.self_attention(text, text, text, text_mask)

        # dropout, residual connection and layer norm
        text = self.self_attn_layer_norm(text + self.dropout(_text))
        # text = [batch size, src len, hid dim]

        # positionwise feedforward
        _text = self.positionwise_feedforward(text)

        # dropout, residual and layer norm
        text = self.ff_layer_norm(text + self.dropout(_text))

        # text = [batch size, src len, hid dim]
        return text


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        # x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        # x = [batch size, seq len, hid dim]

        return x


class Transformer(nn.Module):
    """
    In this class we implement encoder of transformer
    """
    def __init__(self,
                 hid_dim,
                 output_size,
                 encoder,
                 src_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.device = device

        self.fully_connected_layers = nn.Linear(
            in_features=self.encoder.max_length*hid_dim, out_features=output_size
        )

    def make_input_mask(self, text):
        # input_batch.size() = [batch_size, input_len]
        text_mask = (text != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # input_mask = [batch_size, 1, 1, input_len]

        return text_mask

    def forward(self, text):
        # itext.size() = [batch_size, input_len]

        text_mask = self.make_input_mask(text)
        # input_mask.size() = [batch_size, 1, 1, input_len]
        enc_output = self.encoder(text, text_mask)
        # enc_output.size() = [batch_size, input_len, hid_dim]

        # enc_output = enc_output.permute(0, 2, 1)
        # # enc_output.size() = [batch_size, hid_dim, input_len]
        #
        # enc_output = nn.MaxPool1d(enc_output.size()[2])(enc_output).squeeze(2)
        # # enc_input.size() = [batch_size, hid_dim]
        enc_output = torch.flatten(enc_output, start_dim=1)
        # enc_input.size() = [batch_size, input_len * hid_dim]
        return self.fully_connected_layers(enc_output)


if __name__ == '__main__':

    HID_DIM = 256
    ENC_LAYERS = 3
    ENC_HEADS = 8
    ENC_PF_DIM = 512
    ENC_DROPOUT = 0.2
    MAX_LEN = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(vocab_size=2000,
                  hid_dim=HID_DIM,
                  n_layers=ENC_LAYERS,
                  n_heads=ENC_HEADS,
                  pf_dim=ENC_PF_DIM,
                  dropout=ENC_DROPOUT,
                  max_length=MAX_LEN,
                  device=DEVICE)

    model = Transformer(hid_dim=HID_DIM, encoder=enc, output_size=2,
                        device=DEVICE, src_pad_idx=1)

    text = torch.rand((64, MAX_LEN))
    model.forward(text.long())