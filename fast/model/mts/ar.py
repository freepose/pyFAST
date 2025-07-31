#!/usr/bin/env python
# encoding: utf-8

"""
    Autoregressive time series model.
"""

import torch
import torch.nn as nn

from ..base import get_activation_cls


class GAR(nn.Module):
    """
         Global autoregression.
         Note: the target dimension is equal to the input dimension, input_vars === output_vars.
        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param bias: if True, adds a learnable bias to the output.
        :param activation: type str, the activation function.
    """

    def __init__(self, input_window_size: int, output_window_size: int = 1, bias: bool = True,
                 activation: str = 'linear'):
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
            :return:
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
        :param activation: type str, the activation function.
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int = 1,
                 activation: str = 'linear'):
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
            :return:
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
        :param activation: type str, the activation function.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 bias: bool = True, activation: str = 'linear'):
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
            :return:
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

        :param input_window_size:  input window size.
        :param output_window_size: output window size.
        :param hidden_size: hidden size.
    """

    def __init__(self, input_window_size: int, output_window_size: int = 1, hidden_size: int = 64):
        super(ANN, self).__init__()
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.hidden_size = hidden_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_window_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.output_window_size)
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x -> (..., input_window_size, input_vars)
            :param x_mask: mask tensor of input tensor, the shape is ``(..., input_window_size, input_vars)``.
        """
        if x_mask is not None:
            x[~x_mask] = 0.

        x = x.transpose(-2, -1) # -> (..., input_vars, input_window_size)
        x = self.fc(x)          # -> (..., input_vars, output_window_size)
        x = x.transpose(-2, -1) # -> (..., output_window_size, input_vars)

        return x
