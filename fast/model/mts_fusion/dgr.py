#!/usr/bin/env python
# encoding: utf-8

"""
    Dual‐grained Representation (DGR).
"""
from typing import Literal

import torch
import torch.nn as nn

from ..mts import TimeSeriesRNN


class DGR(nn.Module):
    """
        Wang Zhijin, Huang Yaohui, He Bingyan.
        Dual‐grained representation for hand, foot, and mouth disease prediction within public health cyber‐physical systems.
        Software: Practice and Experience, 2021, 51(11): 2290-2305.
        Author: Zhijin Wang

        Dual‐grained Representation (DGR).

        :param input_window_size: input window size of target variable and exogenous variables.
        :param input_vars: number of input variables.
        :param output_window_size: output window size, a.k.a., prediction length.
        :param ex_retain_window_size: input window size of exogenous variables.
        :param ex_vars: number of exogenous variables.
        :param rnn_cls: rnn, lstm, gru.
        :param hidden_size: hidden size of rnn.
        :param ex_hidden_size: hidden size of exogenous rnn.
        :param num_layers: number of rnn layers.
        :param bidirectional: whether to use bidirectional rnn or not.
        :param dropout_rate: dropout rate.
        :param decoder_way: the decoder way is in ['inference', 'mapping'].
    """
    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = 1,
                 rnn_cls: Literal['rnn', 'lstm', 'gru', 'minlstm'] = 'gru',
                 hidden_size: int = 32, ex_hidden_size: int = 32, num_layers: int = 1, bidirectional: bool = False,
                 dropout_rate: float = 0., decoder_way: Literal['inference', 'mapping'] = 'inference'):
        super(DGR, self).__init__()
        self.input_window_size = input_window_size
        self.input_size = input_vars
        self.output_window_size = output_window_size
        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        self.rnn_cls = rnn_cls
        self.hidden_size = hidden_size
        self.exogenous_hidden_size = ex_hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.decoder_way = decoder_way

        # _range = 1.0 / math.sqrt(self.exogenous_input_size + 1)   # = 1.0 / math.sqrt(self.input_size)
        gain = 0.05

        self.weight = nn.Parameter(torch.rand(self.ex_retain_window_size, self.ex_vars) * gain)
        self.bias = nn.Parameter(torch.zeros(ex_vars))

        self.gru_x = TimeSeriesRNN(self.input_size, self.output_window_size, self.input_size,
                                   self.rnn_cls, self.hidden_size, self.num_layers, self.bidirectional,
                                   self.dropout_rate, self.decoder_way)

        self.gru_ex = TimeSeriesRNN(self.ex_vars, self.output_window_size, self.ex_vars,
                                    self.rnn_cls, self.exogenous_hidden_size, self.num_layers, self.bidirectional,
                                    self.dropout_rate, self.decoder_way)

        self.l1 = nn.Linear(self.input_size + self.ex_vars, self.input_size)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: slicing window of the target variable. [batch_size, window_size, input_size]
            :param ex: slicing window of exogenous variables.
                      [batch_size, ex_retain_window_size, ex_vars],
                   or [batch_size, output_window_size, ex_vars],
        """

        out_ex = ex[:, -self.ex_retain_window_size:]
        out_ex = out_ex * self.weight + self.bias
        out_ex = out_ex.softmax(dim=0)    # -> [batch_size, ex_retain_window_size, ex_vars]

        out_ex = self.gru_ex(out_ex)       # -> [batch_size, output_window_size, ex_vars]

        # y = torch.softmax(y, dim=0)
        out_x = self.gru_x(x)       # -> [batch_size, output_window_size, input_size]

        out = torch.cat([out_ex, out_x], dim=2)   # ->[batch_size, output_window_size, input_size + ex_vars]
        # r_xy = torch.softmax(r_xy, dim=2)

        out = self.l1(out)
        return out
