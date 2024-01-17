import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class BiGRUSummarizer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiGRUSummarizer, self).__init__()
        self.bigru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, token_embeddings):
        _, hidden = self.bigru(token_embeddings)

        fwd_hidden = hidden[0:1]  # Forward hidden state
        bwd_hidden = hidden[1:2]  # Backward hidden state
        output_embedding = torch.cat((fwd_hidden, bwd_hidden), dim=2).squeeze(0)

        return output_embedding


# class TransformerSummarizerWithCLS(nn.Module):
#     def __init__(self, input_size, nhead, num_layers, hidden_size, num_positions):
#         super(TransformerSummarizerWithCLS, self).__init__()
#
#         self.pos_encoder = self.positional_encoding(input_size, num_positions)  # +1 for the [CLS] token
#         encoder_layers = TransformerEncoderLayer(input_size, nhead, hidden_size)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
#
#     def forward(self, token_embeddings):
#         batch_size = token_embeddings.shape[0]
#         cls_token = self.cls_embedding.expand(batch_size, -1, -1)  # Create [CLS] tokens for each item in the batch
#         token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)  # Prepend [CLS] tokens to the input
#
#         seq_len = token_embeddings.shape[1]
#         pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
#         pos_embeddings = self.embedding(pos)
#         token_embeddings = token_embeddings + pos_embeddings
#
#         token_embeddings = token_embeddings.transpose(0, 1)  # TransformerEncoder expects (S, N, E) format
#         output = self.transformer_encoder(token_embeddings)
#         cls_embedding = output[0]  # [CLS] embedding is at position 0
#         return cls_embedding
#
#     @staticmethod
#     def positional_encoding(input_size, num_positions):
#         # sinusoidal PE
#         pe = torch.zeros(num_positions, input_size)
#         position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, input_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_size))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         return nn.Parameter(pe, requires_grad=False)

class TransformerSummarizer(nn.Module):

    def __init__(self, input_size, nhead, num_layers):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(input_size, dropout, max_len=num_positions)
        encoder_layers = TransformerEncoderLayer(input_size, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.norm = nn.LayerNorm(input_size)


    def forward(self, x):
        # seq_len = x.shape[1] # make sure to unsqueeze so batch size = 1
        # pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        out = self.transformer_encoder(x.unsqueeze(0))
        # print(out)
        # print(out.shape, "incls")
        out = self.norm(out)
        # print(out.shape, "after cls norm")
        return torch.mean(out, dim=1)

