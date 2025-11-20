#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Literal
from ....mts.rnn import MinLSTM


class SparseNARXRNN(nn.Module):
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
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1, ex_vars: int = 1,
                 rnn_cls: Literal['rnn', 'lstm', 'gru'] = 'gru',
                 hidden_size: int = 32, num_layers: int = 1, bidirectional: bool = False,
                 dropout_rate: float = 0.):
        super(SparseNARXRNN, self).__init__()

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

    def forward(self, x: torch.Tensor, ex: torch.Tensor, ex_mask: torch.Tensor):
        """
            :param x: (batch_size, input_window_size, input_vars)
            :param ex: (batch_size, input_window_size, ex_vars)
            :param ex_mask: (batch_size, input_window_size, ex_vars)
            :return: (batch_size, output_window_size, input_vars)
        """
        if ex_mask is not None:
            ex[~ex_mask] = 0.0

        combined_input = torch.cat((x, ex), dim=2)  # -> (batch_size, input_window_size, input_vars + ex_vars)

        rnn_out, _ = self.rnn(combined_input)  # -> (batch_size, input_window_size, hidden_size)

        out = self.fc(rnn_out)  # -> (batch_size, input_window_size, input_vars)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, hidden_size)

        return out