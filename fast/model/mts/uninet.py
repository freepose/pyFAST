#!/usr/bin/env python
# encoding: utf-8

from typing import Literal

import torch
import torch.nn as nn

from ..base.activation import get_activation_cls
from ..mts import EncoderDecoder
from ..base import DirectionalRepresentation


class DynamicTanh(nn.Module):
    def __init__(self, dim: int, init_alpha: float = 1.):
        super(DynamicTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


class MLPwithDyT(nn.Module):

    def __init__(self, input_dim: int, hidden_units: list, output_dim: int = 1,
                 activation: str = None, dropout_rate: float = 0.):
        super(MLPwithDyT, self).__init__()
        self.input_dim = input_dim  # input feature dimension
        self.hidden_units = hidden_units  # hidden units for each layer
        self.output_dim = output_dim  # output feature dimension
        self.activation = activation  # name of the activation function
        self.dropout_rate = dropout_rate

        units = [self.input_dim, *self.hidden_units, self.output_dim]

        self.mlp = nn.Sequential()
        for i in range(len(units) - 1):
            self.mlp.add_module(f'layer_{i}', nn.Linear(units[i], units[i + 1]))

            if i < len(units) - 2:
                self.mlp.add_module(f'dynamic_tanh_{i}', DynamicTanh(units[i + 1]))
                if self.activation is not None or self.activation != 'linear':
                    self.mlp.add_module(f'activation_{i}', get_activation_cls(self.activation)())
                if self.dropout_rate > 0:
                    self.mlp.add_module(f'dropout_{i}', nn.Dropout(p=self.dropout_rate))

    def forward(self, x: torch.Tensor):
        out = self.mlp(x)  # shape: (batch_size, output_dim)
        return out


class UniNet(nn.Module):
    """
        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of input variables.
        :param rnn_cls: type of RNN model ('rnn', 'lstm', 'gru', 'minlstm').
        :param hidden_size: number of features in the hidden state.
        :param num_layers: number of recurrent layers.
        :param bidirectional: whether to use bidirectional RNN or not.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 rnn_cls: Literal['gru', 'lstm'] = 'lstm', hidden_size: int = 512, num_layers: int = 2,
                 bidirectional: bool = True, dropout_rate: float = 0.05):
        super(UniNet, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate if num_layers > 1 else 0.

        self.ed = EncoderDecoder(input_vars, output_window_size, output_vars,
                                 rnn_cls, hidden_size, num_layers, bidirectional, dropout_rate, decoder_way='mapping')

        self.dr0 = DirectionalRepresentation(output_window_size, output_vars, 0, 'gelu', dropout_rate)
        self.dr1 = DirectionalRepresentation(output_window_size, output_vars, 1, 'gelu', dropout_rate)

        self.mlp1 = MLPwithDyT(output_vars * 3, [], output_vars)
        self.w1 = nn.Parameter(torch.rand((output_window_size, output_vars)), )

        self.dr3 = DirectionalRepresentation(output_window_size, output_vars, 0, 'gelu', dropout_rate)
        self.dr4 = DirectionalRepresentation(output_window_size, output_vars, 0, 'gelu', dropout_rate)

        self.mlp2 = MLPwithDyT(output_vars * 3, [], output_vars)
        self.w2 = nn.Parameter(torch.rand((output_window_size, output_vars)), )

    def forward(self, x: torch.Tensor):
        """
            x: (batch_size, input_window_size, input_vars)
        """

        out = self.ed(x)
        out0 = self.dr0(out)
        out1 = self.dr1(out)

        cout = torch.cat([out0, out1, out], dim=2)
        out = self.mlp1(cout) + out * self.w1

        out0 = self.dr3(out)
        out1 = self.dr4(out)

        cout = torch.cat([out0, out1, out], dim=2)
        out = self.mlp2(cout) + out * self.w2

        return out
