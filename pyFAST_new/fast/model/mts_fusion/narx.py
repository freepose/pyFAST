#!/usr/bin/env python
# encoding: utf-8

"""

    NARX: Nonlinear AutoRegressive model with exogenous inputs.

"""
from typing import Literal

import torch
import torch.nn as nn

from ..base.mlp import MLP
from ..mts.ar import AR, VAR
from ..mts.rnn import MinLSTM


class ARX(nn.Module):
    """
        AutoRegressive model with exogenous inputs.
        :param input_window_size: input window size, a.k.a. ``p``, lag of the autoregressive model.
        :param input_vars: number of input variables.
        :param output_window_size: output window size, a.k.a. ``h``, prediction length.
        :param ex_retain_window_size: retain window size of exogenous factors, a.k.a. ``q``, lag of the exogenous model.
        :param ex_vars: number of exogenous input variables.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, out_vars: int = 1,
                 ex_retain_window_size: int = None, ex_vars: int = 1):
        super(ARX, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_retain_window_size = input_window_size if ex_retain_window_size is None else ex_retain_window_size
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size
        self.out_vars = out_vars

        assert self.input_window_size >= self.ex_retain_window_size, 'input_window_size >= ex_input_window'

        self.ar = VAR(self.input_window_size, self.input_vars, self.output_window_size, self.out_vars, False)
        self.ex_ar = VAR(self.ex_retain_window_size, self.ex_vars, self.output_window_size, self.out_vars)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
        :param x: shape is (batch_size, input_window_size, input_vars)
        :param ex: shape is (batch_size, input_window_size, ex_vars)
        :return: shape is (batch_size, output_window_size, out_vars)
        """
        x_out = self.ar(x)

        ex = ex[:, -self.ex_retain_window_size:, :]  # -> (batch_size, ex_retain_window_size, ex_vars)
        ex_out = self.ex_ar(ex)

        out = x_out + ex_out

        return out


class NARXMLP(nn.Module):
    """
        NARX: Nonlinear AutoRegressive model with exogenous inputs.

        :param input_window_size: input window size.
        :param input_vars: number of input variables. ``input_vars`` == ``output_vars``
        :param ex_retain_window_size: retain window size of exogenous factors.
        :param ex_vars: number of exogenous input variables.
        :param output_window_size: output window size.
        :param hidden_units: hidden size of mlp.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 ex_retain_window_size: int = None, ex_vars: int = 1,
                 hidden_units: list[int] = [32], use_layer_norm: bool = False, activation: str = 'linear'):
        super(NARXMLP, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_input_window_size = input_window_size if ex_retain_window_size is None else ex_retain_window_size
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size
        self.hidden_units = hidden_units

        assert input_window_size >= ex_retain_window_size, 'input_window_size >= ex_input_window'

        input_dim = input_window_size * input_vars + ex_retain_window_size * ex_vars
        self.mlp = MLP(input_dim, hidden_units, output_window_size * input_vars, use_layer_norm, activation, 0)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: (batch_size, input_window_size, input_vars)
            :param ex: (batch_size, input_window_size, ex_vars)
            :return: (batch_size, output_window_size, input_vars)
        """
        ex = ex[:, -self.ex_input_window_size:, :]  # -> (batch_size, ex_input_window_size, ex_vars)

        batch_size = x.size(0)

        x = x.flatten(1)  # -> (batch_size, input_window_size * input_vars)
        ex = ex.flatten(1)  # -> (batch_size, ex_input_window_size * ex_vars)
        out = torch.cat((x, ex), dim=1)

        out = self.mlp(out)
        out = out.view(batch_size, self.output_window_size, self.input_vars)

        return out


class NARXRNN(nn.Module):
    """
        NARX-RNN: Nonlinear AutoRegressive model with exogenous inputs using RNN.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param ex_vars: number of exogenous input variables.
        :param output_window_size: output window size.
        :param rnn_cls: rnn, lstm, gru.
        :param hidden_size: hidden size of rnn.
        :param num_layers: number of rnn layers.
        :param bidirectional: whether to use bidirectional rnn or not.
        :param dropout_rate: dropout rate.
        :param use_layer_norm: whether to use layer norm or not.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1, ex_vars: int = 1,
                 rnn_cls: Literal['rnn', 'lstm', 'gru'] = 'gru',
                 hidden_size: int = 32, num_layers: int = 1, bidirectional: bool = False,
                 dropout_rate: float = 0.):
        super(NARXRNN, self).__init__()

        assert input_window_size >= output_window_size, 'input_window_size >= output_window_size'

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate if num_layers > 1 else 0.

        rnn_cls_dict = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'minlstm': MinLSTM}
        model_cls = rnn_cls_dict.get(rnn_cls)

        self.rnn = model_cls(input_vars + ex_vars, self.hidden_size, self.num_layers, batch_first=True,
                             bidirectional=self.bidirectional, dropout=self.dropout_rate)

        self.fc = nn.Linear(self.hidden_size, self.input_vars)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: (batch_size, input_window_size, input_vars)
            :param ex: (batch_size, input_window_size, ex_vars)
            :return: (batch_size, output_window_size, input_vars)
        """

        combined_input = torch.cat((x, ex), dim=2)  # -> (batch_size, input_window_size, input_vars + ex_vars)

        rnn_out, _ = self.rnn(combined_input)  # -> (batch_size, input_window_size, hidden_size)

        out = self.fc(rnn_out)  # -> (batch_size, input_window_size, input_vars)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, hidden_size)

        return out
