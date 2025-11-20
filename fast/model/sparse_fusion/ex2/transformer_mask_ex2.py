#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Literal
from ....model.mts.transformer.embedding import TokenEmbedding, PositionalEncoding


class TransformerMaskEx2(nn.Module):
    """
        Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I.
        'Attention Is All You Need', NeurIPS 2017.
        url: https://arxiv.org/abs/1706.03762.

        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param ex2_vars: number of exogenous2 (a.k.a., pre-known) variables.
        :param label_window_size: label window is intersections between input and output windows.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param num_decoder_layers: number of decoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_vars: int, output_vars: int = 1, ex2_vars: int = None,
                 label_window_size: int = None,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 1,
                 num_decoder_layers: int = 1,
                 dim_ff: int = 2048,
                 dropout_rate: float = 0.,
                 activation: Literal['relu', 'gelu'] = 'relu'):
        super(TransformerMaskEx2, self).__init__()

        if label_window_size is not None:
            assert 0 < label_window_size, 'Invalid window parameters.'

        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_vars = input_vars
        self.input_window_size = None   # to be set/update in forward()
        self.output_window_size =  None  # to be set/update in forward()
        self.output_vars = output_vars
        self.label_window_size = label_window_size
        self.ex2_vars = ex2_vars

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.activation = activation

        self.encoder_value_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.encoder_position_embedding = PositionalEncoding(self.d_model)
        if self.ex2_vars:
            self.encoder_temporal_embedding = nn.Linear(self.ex2_vars, self.d_model, bias=False)
        self.encoder_dropout = nn.Dropout(dropout_rate)

        self.decoder_value_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.decoder_position_embedding = PositionalEncoding(self.d_model)
        if self.ex2_vars:
            self.decoder_temporal_embedding = nn.Linear(self.ex2_vars, self.d_model, bias=False)
        self.decoder_dropout = nn.Dropout(dropout_rate)

        self.transformer = nn.Transformer(self.d_model, self.num_heads,
                                          self.num_encoder_layers, self.num_decoder_layers, self.dim_ff,
                                          dropout_rate, batch_first=True)

        self.fc = nn.Linear(self.d_model, self.output_vars)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, ex2: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input window time series, shape is (batch, input_window_size, input_vars).
            :param ex2: [Optional] temporal features for the input and output windows,
                        shape is (batch, input_window_size + output_window_size, temporal_vars).
                        The time features are **preknown** information.

                        If the temporal features are not provided, then works as a standard Transformer model.
            :return: output window time series,
                     shape is (batch, output_window_size <= input_window_size, output_vars == input_vars).
        """
        self.label_window_size = self.input_window_size // 2

        xe = x
        if x_mask is not None:
            xe[~x_mask] = 0  # set nan values to zeros.

        if ex2 is not None:
            ex2_inputs = ex2[:, :-self.output_window_size, :]
            ex2_outputs = ex2[:, -self.label_window_size - self.output_window_size:, :]

        x_embedding = self.encoder_value_embedding(xe) + self.encoder_position_embedding(xe) # -> (batch_size, input_window_size, d_model)
        if self.ex2_vars and ex2 is not None:
            x_embedding += self.encoder_temporal_embedding(ex2_inputs)
        x_embedding = self.encoder_dropout(x_embedding)

        # target -> (batch_size, label_window_size + output_window_size, input_vars)
        batch_size, seq_len, input_vars = x.shape
        target_shape = (batch_size, self.label_window_size + self.output_window_size, input_vars)
        target = torch.zeros(*target_shape, dtype=x.dtype, device=x.device)
        target[:, :self.label_window_size, :] = xe[:, -self.label_window_size:, :]

        # target_embedding -> (batch_size, label_window_size + output_window_size, d_model)
        target_embedding = self.decoder_value_embedding(target) + self.decoder_position_embedding(target)
        if self.ex2_vars and ex2 is not None:
            target_embedding += self.decoder_temporal_embedding(ex2_outputs)
        target_embedding = self.decoder_dropout(target_embedding)

        out = self.transformer(src=x_embedding, tgt=target_embedding) # -> (batch_size, label_window_size + output_window_size, d_model)
        out = self.fc(out)  # -> (batch_size, label_window_size + output_window_size, output_vars)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, output_vars)

        return out
