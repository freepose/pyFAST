#!/usr/bin/env python
# encoding: utf-8
from typing import Literal

import torch
import torch.nn as nn

from .embedding import TokenEmbedding, PositionalEncoding


class Transformer(nn.Module):
    """
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.
        'Attention Is All You Need', NeurIPS 2017.
        url: https://arxiv.org/abs/1706.03762.

        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param label_window_size: label window is intersections between input and output windows.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param num_decoder_layers: number of decoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_vars: int, output_window_size: int = 96, output_vars: int = 1,
                 label_window_size: int = 0,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1,
                 dim_ff: int = 2048,
                 dropout_rate: float = 0.,
                 activation: Literal['relu', 'gelu'] = 'relu'):
        super(Transformer, self).__init__()
        assert 0 <= label_window_size, 'Invalid window parameters.'
        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.label_window_size = label_window_size

        self.d_model = d_model  # model dimension, a.k.a., embedding size
        self.num_heads = num_heads  # head number, a.k.a, attention mechanism number
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.activation = activation

        self.encoder_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.encoder_pe = PositionalEncoding(self.d_model)
        self.encoder_dropout = nn.Dropout(dropout_rate)

        self.decoder_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.decoder_pe = PositionalEncoding(self.d_model)
        self.decoder_dropout = nn.Dropout(dropout_rate)

        self.transformer = nn.Transformer(self.d_model, self.num_heads,
                                          self.num_encoder_layers, self.num_decoder_layers, self.dim_ff,
                                          dropout_rate, batch_first=True)

        self.fc = nn.Linear(self.d_model, self.output_vars)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        """
            :param x: input window time series, shape is (batch, input_window_size, input_vars).
            :param x_mask: mask for input, shape is (batch, input_window_size, input_vars).
            :return: output window time series,
                     shape is (batch, output_window_size <= input_window_size, output_vars == input_vars).
        """
        xe = x
        if x_mask is not None:
            xe[~x_mask] = 0    # set nan values to zeros.

        x_embedding = self.encoder_embedding(xe) + self.encoder_pe(xe)  # -> (batch_size, input_window_size, d_model)
        x_embedding = self.encoder_dropout(x_embedding)

        # provide context to the decoder
        batch_size, seq_len, input_vars = x.shape
        target_shape = (batch_size, self.label_window_size + seq_len, input_vars)
        target = torch.zeros(*target_shape, dtype=x.dtype, device=x.device)
        target[:, :self.label_window_size, :] = xe[:, seq_len-self.label_window_size:, :]  # intersection (label) window

        # -> (batch_size, label_window_size + output_window_size, d_model)
        target_embedding = self.decoder_embedding(target) + self.decoder_pe(target)
        target_embedding = self.decoder_dropout(target_embedding)

        out = self.transformer(src=x_embedding, tgt=target_embedding, src_mask=x_mask)
        out = self.fc(out)  # -> (batch_size, label_window_size + output_window_size, d_model)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, input_vars)

        return out
