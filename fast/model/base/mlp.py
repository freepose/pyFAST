#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Literal
from .activation import get_activation_cls
from .norm import DynamicTanh


class MLP(nn.Module):
    """
        Multiple Layer Perceptron (MLP).

        :param input_dim: input feature dimension.
        :param hidden_units: list of positive integer, the layer number and units in each layer.
        :param output_dim: output feature dimension.
        :param layer_norm: layer normalization in ('DyT' or 'LN'). If None, no layer normalization is applied.
        :param activation: the activation function to use.
        :param dropout_rate: float in [0, 1). Fraction of the units to dropout.
    """
    def __init__(self, input_dim: int, hidden_units: list, output_dim: int = 1,
                 layer_norm: Literal['DyT', 'LN']  = None, activation: str = None, dropout_rate: float = 0.):
        super(MLP, self).__init__()
        self.input_dim = input_dim  # input feature dimension
        self.hidden_units = hidden_units  # hidden units for each layer
        self.output_dim = output_dim  # output feature dimension
        self.layer_norm = layer_norm
        self.activation = activation  # name of the activation function
        self.dropout_rate = dropout_rate

        units = [self.input_dim, *self.hidden_units, self.output_dim]

        self.mlp = nn.Sequential()
        for i in range(len(units) - 1):
            self.mlp.add_module(f'layer_{i}', nn.Linear(units[i], units[i + 1]))

            if i < len(units) - 2:
                if self.layer_norm is not None:
                    if self.layer_norm == 'DyT':
                        self.mlp.add_module(f'dynamic_tanh_{i}', DynamicTanh(units[i + 1]))
                    elif self.layer_norm == 'LN':
                        self.mlp.add_module(f'layer_norm_{i}', nn.LayerNorm(units[i + 1]))
                if self.activation is not None or self.activation != 'linear':
                    self.mlp.add_module(f'activation_{i}', get_activation_cls(self.activation)())
                if self.dropout_rate > 0:
                    self.mlp.add_module(f'dropout_{i}', nn.Dropout(p=self.dropout_rate))

    def forward(self, x: torch.Tensor):
        """
            x, the input tensor, shape: ``(batch_size, input_dim)``
        """
        out = self.mlp(x)  # shape: (batch_size, output_dim)
        return out
