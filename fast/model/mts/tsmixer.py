#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from ...data import InstanceStandardScale
from .ar import GAR, AR


class ResBlock(nn.Module):
    """
        Residual block of TSMixer.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1,
                 hidden_size: int = 2048, dropout_rate: float = 0.0):
        super(ResBlock, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.hidden_size = hidden_size

        self.temporal = nn.Sequential(
            nn.Flatten(1, 2),  # -> (batch_size, input_window_size * input_vars)
            nn.BatchNorm1d(self.input_window_size * self.input_vars),
            nn.Unflatten(1, (self.input_window_size, self.input_vars)),
            # -> (batch_size, input_window_size, input_vars)
            GAR(self.input_window_size, self.input_window_size),
            nn.Dropout(dropout_rate),
        )

        self.feature = nn.Sequential(
            nn.Flatten(1, 2),  # -> (batch_size, input_vars * input_window_size)
            nn.BatchNorm1d(self.input_vars * self.input_window_size),
            nn.Unflatten(1, (self.input_window_size, self.input_vars)),
            # -> (batch_size, input_window_size, input_vars)
            nn.Linear(self.input_vars, self.hidden_size),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, self.input_vars),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        """
            :param x: Input tensor of shape (batch_size, input_window_size, input_vars).
        """

        _temporal = self.temporal(x)
        _feature = self.feature(x)

        return x + _temporal + _feature


class TSMixer(nn.Module):
    """
        TSMixer: An All-MLP Architecture for Time Series Forecasting
        Si-An Chen, Chun-Liang Li, Nate Yoder, Sercan O. Arik, Tomas Pfister
        Transactions on Machine Learning Research (09/2023)
        url: https://arxiv.org/abs/2303.06053

        Author provided code: https://github.com/giuliomattolin/tsmixer-pytorch

        :param input_window_size: input sequence length.
        :param input_vars: number of input variables (channels/features).
        :param output_window_size: output sequence length.
        :param num_blocks: number of residual blocks.
        :param block_hidden_size: hidden size of the residual blocks.
        :param dropout_rate: dropout rate applied to the input.
        :param use_instance_scale: whether to use instance standard scaling (i.e., RevIN).
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1,
                 num_blocks: int = 1, block_hidden_size: int = 2048, dropout_rate: float = 0.0,
                 use_instance_scale: bool = True):
        super(TSMixer, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.num_blocks = num_blocks
        self.block_hidden_size = block_hidden_size
        self.use_instance_scale = use_instance_scale

        self.res_blocks = nn.ModuleList(
            [ResBlock(self.input_window_size, self.input_vars, self.block_hidden_size, dropout_rate)
             for _ in range(num_blocks)]
        )

        self.l1 = GAR(self.input_window_size, self.output_window_size)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x):
        """
            :param x: Input tensor of shape (batch_size, input_window_size, input_vars).
        """

        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x)

        for res_block in self.res_blocks:
            x = res_block(x)  # -> (batch_size, input_window_size, input_vars)

        x = self.l1(x)  # -> (batch_size, output_window_size, input_vars)

        if self.use_instance_scale:
            x = self.inst_scaler.inverse_transform(x)

        return x
