#!/usr/bin/env python
# encoding: utf-8

from typing import Literal

import torch
import torch.nn as nn

from ..base import MLP
from ..mts import EncoderDecoder
from ..base import DirectionalRepresentation


class AFTS(nn.Module):
    """
        Aligned Feature Time Series (AFTS) model.

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
        super(AFTS, self).__init__()
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

        self.mlp1 = MLP(output_vars * 3, [], output_vars)
        self.w1 = nn.Parameter(torch.rand((output_window_size, output_vars)))

        self.dr3 = DirectionalRepresentation(output_window_size, output_vars, 0, 'gelu', dropout_rate)
        self.dr4 = DirectionalRepresentation(output_window_size, output_vars, 0, 'gelu', dropout_rate)

        self.mlp2 = MLP(output_vars * 3, [], output_vars)
        self.w2 = nn.Parameter(torch.rand((output_window_size, output_vars)))

    def forward(self, x: torch.Tensor):
        """
            :param x: shape is ``(batch_size, input_window_size, input_vars)``

        """

        out = self.ed(x)
        out0 = self.dr0(out)
        out1 = self.dr1(out)

        c_out = torch.cat([out0, out1, out], dim=2)
        out = self.mlp1(c_out) + out * self.w1

        out0 = self.dr3(out)
        out1 = self.dr4(out)

        c_out = torch.cat([out0, out1, out], dim=2)
        out = self.mlp2(c_out) + out * self.w2

        return out
