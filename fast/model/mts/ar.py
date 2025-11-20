#!/usr/bin/env python
# encoding: utf-8

"""
    Autoregressive time series model.
"""

import torch
import torch.nn as nn

from typing import Literal, Union, List, Tuple
from ..base import get_activation_cls, ActivationName
from ..base import MLP


class GAR(nn.Module):
    """
        Global autoregression.
        Note: the target dimension is equal to the input dimension, input_vars === output_vars.

        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param bias: if True, adds a learnable bias to the output.
        :param activation: type str, one in ['linear', 'relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'sin'].
    """

    def __init__(self, input_window_size: int, output_window_size: int = 1, bias: bool = True,
                 activation: ActivationName = 'linear'):
        super(GAR, self).__init__()
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.activation = activation

        self.l1 = nn.Linear(self.input_window_size, self.output_window_size, bias)
        self.activate = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :param x_mask: mask tensor of input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :return: output tensor, the shape is ``(..., output_window_size, input_vars)``.
        """

        if x_mask is not None:
            x[~x_mask] = 0.

        x = x.transpose(-1, -2)  # -> (..., input_vars, input_window_size)
        x = self.l1(x)  # -> (..., input_vars, output_window_size)
        x = x.transpose(-1, -2)  # x -> (..., output_window_size, input_vars)

        x = self.activate(x)

        return x


class AR(nn.Module):
    """
        Autoregressive model.
        The idea: the future events are linear combinations of past dataset points of oneself.
        Note: the target dimension is equal to the input dimension, input_vars === output_vars.
        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param activation: type str, one in ['linear', 'relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'sin'].
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int = 1,
                 activation: ActivationName = 'linear'):
        super(AR, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.activation = activation

        self.weight = nn.Parameter(torch.randn(self.input_window_size,
                                               self.input_vars,
                                               self.output_window_size))
        self.bias = nn.Parameter(torch.randn(self.input_vars))

        self.activate = get_activation_cls(activation)()

        nn.init.xavier_uniform_(self.weight, gain=0.01)
        nn.init.constant_(self.bias, 0.)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :param x_mask: mask tensor of input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :return: output tensor, the shape is ``(..., output_window_size, input_vars)``.
        """

        if x_mask is not None:
            x[~x_mask] = 0.

        x = torch.einsum('...jk,jkl->...lk', x, self.weight)  # x -> (batch_size, output_window_size, input_vars)
        x = x + self.bias  # x -> (batch_size, output_window_size, input_vars)

        x = self.activate(x)

        return x


class VAR(nn.Module):
    """
        Vector Autoregressive model.
        The idea: the future events are linear combinations of past dataset points of all inputs.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param activation: type str, one in ['linear', 'relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'sin'].
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 bias: bool = True, activation: ActivationName = 'linear'):
        super(VAR, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.activation = activation

        self.l1 = nn.Linear(self.input_window_size * self.input_vars, self.output_window_size * self.output_vars, bias)

        self.activate = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :param x_mask: mask tensor of input tensor, the shape is ``(..., input_window_size, input_vars)``.
            :return: output tensor, the shape is ``(..., output_window_size, output_vars)``.
        """

        if x_mask is not None:
            x[~x_mask] = 0.

        x = x.flatten(-2, -1)  # x -> (..., window_size * input_vars)
        x = self.l1(x)  # -> (..., output_window_size * output_vars)
        x = x.view(-1, self.output_window_size, self.output_vars)  # => (..., output_window_size * output_vars)

        x = self.activate(x)

        return x


class ANN(nn.Module):
    """
        Yiyu Ding, Thomas Ohlson Timoudas, Qian Wang, Shuqin Chen, Helge Brattebø, Natasa Nord.
        A study on data-driven hybrid heating load prediction methods in low-temperature
          district heating: An example for nursing homes in Nordic countries.
        Energy Conversion and Management 2022.
        url: https://doi.org/10.1016/j.enconman.2022.116163

        The default ``hidden_sizes`` is [64], ``activation`` is 'relu',
        which is the same as the original paper.

        We extend the ``hidden_sizes`` to be a list of integers, the choice of activation functions.

        :param input_window_size:  input window size.
        :param output_window_size: output window size.
        :param hidden_sizes: hidden layer sizes, can be a single integer or a list of integers.
        :param layer_norm: type str, the layer normalization method, can be 'DyT' or 'LN'.
                           If None, no layer normalization is applied.
        :param activation: type str, one in ['linear', 'relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'sin'].
        :param dropout_rate: float in [0, 1), the dropout rate to apply after each layer.
                             If 0, no dropout is applied.
    """

    def __init__(self, input_window_size: int, output_window_size: int = 1,
                 hidden_sizes: Union[int, Tuple[int, ...], List[int]] = 64,
                 layer_norm: Literal['DyT', 'LN'] = None,
                 activation: ActivationName = 'relu',
                 dropout_rate: float = 0.):
        super(ANN, self).__init__()

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, (list, tuple)) else [hidden_sizes]
        self.layer_norm = layer_norm
        self.activation = activation
        self.dropout_rate = dropout_rate

        self.mlp = MLP(self.input_window_size, self.hidden_sizes, self.output_window_size,
                       self.layer_norm, self.activation, self.dropout_rate)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x -> (..., input_window_size, input_vars)
            :param x_mask: mask tensor of input tensor, the shape is ``(..., input_window_size, input_vars)``.
                           If None, no mask is applied.
            :return: output tensor, the shape is ``(..., output_window_size, input_vars)``.
        """
        if x_mask is not None:
            x[~x_mask] = 0.

        x = x.transpose(-2, -1)  # -> (..., input_vars, input_window_size)
        x = self.mlp(x)  # -> (..., input_vars, output_window_size)
        x = x.transpose(-2, -1)  # -> (..., output_window_size, input_vars)

        return x
