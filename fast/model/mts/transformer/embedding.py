#!/usr/bin/env python
# encoding: utf-8
import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
        Embedding for Transformer.
    """

    def __init__(self, in_channels, embedding_size):
        super(TokenEmbedding, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels, embedding_size, 3,
                                   padding=padding, padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size, input_size] """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x  # -> [batch_size, input_window_size, embedding_size]


class PositionalEncoding(nn.Module):
    """
        Positional encoding for Transformer.
        :param d_model: model dimension, a.k.a., embedding size.
        :param max_seq_len: maximum sequence length.
    """

    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        self.pe = nn.Parameter(torch.zeros(max_seq_len, d_model), requires_grad=False)

        position = torch.arange(0, max_seq_len).unsqueeze(1)  # -> (max_seq_len, 1)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).exp()  # -> (d_model // 2)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """ x -> (batch_size, input_window_size, input_vars) """
        return self.pe[:x.size(1)].unsqueeze(0)  # -> (1, input_window_size, d_model)
