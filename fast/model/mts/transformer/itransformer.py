#!/usr/bin/env python
# encoding: utf-8
from typing import Literal

import torch
import torch.nn as nn

from ....data import InstanceScale, InstanceStandardScale


class iTransformer(nn.Module):
    """
        Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, Mingsheng Long
        iTransformer: Inverted Transformers Are Effective for Time Series Forecasting, ICLR 2024
        Link: https://arxiv.org/abs/2310.06625
        Code: https://github.com/thuml/iTransformer

        Encoder-only transformer. Attention mechanism is applied to time dimension (i.e., time steps).
        The sequence length of input window should be fixed.

        :param input_window_size: output window size.
        :param input_vars: number of input variables. ``output_vars`` == ``input_vars``.
        :param output_window_size: output window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
        :param activation: activation function name.
        :param use_instance_scale: whether to use instance standard scale (a.k.a., RevIN).
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 1,
                 dim_ff: int = 2048,
                 dropout_rate: float = 0.,
                 activation: Literal['relu', 'gelu'] = 'relu',
                 use_instance_scale: bool = False):
        super(iTransformer, self).__init__()
        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dim_ff = dim_ff

        self.seq_embedding = nn.Linear(self.input_window_size, self.d_model)
        self.dropout = nn.Dropout(dropout_rate)

        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.num_heads, self.dim_ff,
                                                        dropout_rate, activation, batch_first=True)
        self.encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.num_encoder_layers, self.encoder_norm)

        self.projection = nn.Linear(self.d_model, self.output_window_size)

        self.inst_scale = InstanceStandardScale() if use_instance_scale else InstanceScale()

    def forward(self, x: torch.Tensor):
        """ x -> (batch_size, input_window_size, input_vars) """

        norm_x = self.inst_scale.fit_transform(x)
        norm_x = norm_x.permute(0, 2, 1)    # -> (batch_size, input_vars, input_window_size)

        x_embedding = self.seq_embedding(norm_x)
        x_embedding = self.dropout(x_embedding)

        encoder_out = self.encoder(x_embedding)
        out = self.projection(encoder_out)

        out = out.permute(0, 2, 1)    # -> (batch_size, output_window_size, input_vars)
        out = self.inst_scale.inverse_transform(out)

        return out
