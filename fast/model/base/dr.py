#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn

from .activation import get_activation_cls


class DirectionalRepresentation(nn.Module):
    """
        Learning directional representation from a windowed time series.
        Element-wise attention mechanism. (Good at short-sequence)
     """

    def __init__(self, window_size: int, input_size: int, r_dim: int = 0,
                 activation: str = 'linear', dropout_rate: float = 0.):
        super(DirectionalRepresentation, self).__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.r_dim = r_dim
        self.activation = get_activation_cls(activation)()  # default as linear

        self.weight = nn.Parameter(torch.zeros(self.window_size, self.input_size))
        nn.init.uniform_(self.weight, a=0.01, b=0.1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: [batch_size, window_size, input_size]
        """
        x = self.activation(x)
        out = x * self.weight  # element-wise linear mapping

        # out = self.activation(out)

        if self.r_dim != -1:
            out = out.softmax(dim=self.r_dim)

        # out = out * x   # element-wise attention, useful for short-sequence: infectious disease cases

        out = self.dropout(out)

        return out
