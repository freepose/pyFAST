#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Optional
from .activation import get_activation_cls, ActivationName


class DirectionalRepresentation(nn.Module):
    """
        Learning directional representation from a windowed time series.
        Element-wise attention mechanism. (Good at short-sequence)

        :param window_size: int, the size of the time window.
        :param input_vars: int, the feature dimension of the input time series.
        :param r_dim: Optional[int], the dimension to apply softmax normalization. Default: None (no softmax).
                        If set to -1, it means the last dimension.
        :param activation: str, the activation function to use. Default: 'linear'.
        :param dropout_rate: float, the dropout rate to apply after the attention mechanism. Default: 0.0.
     """

    def __init__(self, window_size: int, input_vars: int, r_dim: Optional[int] = None,
                 activation: ActivationName = 'linear', dropout_rate: float = 0.):
        super(DirectionalRepresentation, self).__init__()

        if r_dim is not None:
            assert r_dim >= -1, "r_dim should be greater than -1, and -1 means the last dimension."

        self.window_size = window_size
        self.input_vars = input_vars
        self.r_dim = r_dim
        self.activation = get_activation_cls(activation)()  # default as linear

        self.weight = nn.Parameter(torch.zeros(self.window_size, self.input_vars))
        nn.init.uniform_(self.weight, a=0.01, b=0.1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: [batch_size, window_size, input_size]
        """
        x = self.activation(x)
        out = x * self.weight  # element-wise linear mapping

        # out = self.activation(out)

        if self.r_dim is not None:
            out = out.softmax(dim=self.r_dim)

        # out = out * x   # element-wise attention, useful for short-sequence: infectious disease cases

        out = self.dropout(out)

        return out
