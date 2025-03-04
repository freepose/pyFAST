#!/usr/bin/env python
# encoding: utf-8

"""

    External Data-Driven Modeling for imputation or forecasting.

"""

from typing import Literal

import torch
import torch.nn as nn
from torch.utils.hipify.hipify_python import mapping

from ..mts import TimeSeriesRNN, EncoderDecoder, Transformer
from ...data import InstanceScale, InstanceStandardScale


class ExDDM(nn.Module):
    """
        External Data-Driven Modeling (For pKa prediction/estimation).
        For pKa estimation, assure that ``input_window_size`` == ``output_window_size``.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param ex_retain_window_size: retain window size of exogenous factors.
        :param ex_vars: number of exogenous input variables.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 ex_retain_window_size: int = None, ex_vars: int = 1):
        super(ExDDM, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_retain_window_size = input_window_size if ex_retain_window_size is None else ex_retain_window_size
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size

        assert self.input_window_size >= self.ex_retain_window_size, 'input_window_size >= ex_input_window'

        self.rnn = TimeSeriesRNN(self.ex_vars, self.output_window_size, self.input_vars, 'lstm',
                                 hidden_size=128, num_layers=1, bidirectional=False, dropout_rate=0.2,
                                 decoder_way='mapping')
        self.trans = Transformer(self.ex_vars, self.output_window_size, self.input_vars, 0, 512, 8, 12, 12, 2048, 0.1)

        # esm (N, feature_dim) -> pK_a (N, 1): GCN, GAT, etc.

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, ex: torch.Tensor):
        """
            :param x: shape is (batch_size, input_window_size, input_vars), data type is float.
            :param x_mask: shape is (batch_size, input_window_size, input_vars), data type is bool.
            :param ex: shape is (batch_size, input_window_size, ex_vars), data type is float.
            :return: shape is (batch_size, output_window_size, output_vars), data type is float.
        """

        out = self.trans(ex)   # -> (batch_size, output_window_size, output_vars)

        return out
