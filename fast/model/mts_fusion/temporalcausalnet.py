#!/usr/bin/env python
# encoding: utf-8

from typing import Literal, List

import torch
import torch.nn as nn

from ..mts import GAR, EncoderDecoder
from ..base import MLP, DirectionalRepresentation


class TemporalCausalNet(nn.Module):
    """
        Temporal Causal Net.
        It applied causal relation architecture to Encoder-Decoder architecture,
        and achieved amazing personalized forecasting performance.

        Author: Senzhen Wu

        :param input_window_size: input window size of target variable.
        :param input_vars: number of input variables.
        :param output_window_size: output window size, a.k.a., prediction length.
        :param output_vars: number of output variables.
        :param ex_retain_window_size: input window size of exogenous variable.
        :param ex_vars: number of exogenous variables.
        :param ex_linear_layers: the layers of the pre-processing MLP of exogenous data.
        :param target_linear_layers: the layers of the pre-processing MLP of target data.
        :param hidden_size: the latent space size of time series data after MLP pre-processing.
        :param rnn_cls: the type of Encoder-Decoder model ('rnn', 'lstm', 'gru', 'minlstm').
        :param rnn_hidden_size: number of features in the hidden state of Encoder-Decoder.
        :param rnn_num_layers: number of recurrent layers of Encoder-Decoder.
        :param rnn_bidirectional: whether to use bidirectional for Encoder-Decoder or not.
        :param dr_ratio: the extent to which the Directional Representation Component refines the data.
        :param origin_ratio: the extent to which the prediction retains the features of the original data.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = None,
                 ex_linear_layers: List = [32, 128], target_linear_layers: List = [32, 128], hidden_size: int = 64,
                 rnn_cls: Literal['rnn', 'lstm', 'gru', 'minlstm'] = 'lstm', rnn_hidden_size: int = 512,
                 rnn_num_layers: int = 3, rnn_bidirectional: bool = True,
                 dr_ratio: float = 0.01, origin_ratio: float = 1):
        super(TemporalCausalNet, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        self.dr_ratio = dr_ratio
        self.origin_ratio = origin_ratio

        self.ex_mlp = MLP(ex_vars, ex_linear_layers, hidden_size)
        self.mlp = MLP(input_vars, target_linear_layers, hidden_size)

        self.weight = nn.Parameter(torch.randn(ex_retain_window_size, hidden_size))
        self.ed = EncoderDecoder(hidden_size, ex_retain_window_size, hidden_size,
                                 rnn_cls, rnn_hidden_size, rnn_num_layers, rnn_bidirectional,
                                 0.01, 'mapping')

        self.dr0 = DirectionalRepresentation(ex_retain_window_size, hidden_size, r_dim=0)
        self.dr1 = DirectionalRepresentation(ex_retain_window_size, hidden_size, r_dim=1)
        self.dr2 = DirectionalRepresentation(ex_retain_window_size, hidden_size, r_dim=2)

        self.weight = nn.Parameter(torch.randn(ex_retain_window_size, hidden_size))
        self.drlin = nn.Linear(hidden_size * 3, output_vars)
        self.endxlin = nn.Linear(hidden_size, output_vars)
        self.gar = GAR(ex_retain_window_size, output_window_size)

    def forward(self, x: torch.Tensor, ex: torch.Tensor = None):
        """
            :param x: Input tensor of shape (batch_size, input_window_size, input_vars).
            :param ex: Exogenous input tensor of shape (batch_size, input_window_size, ex_vars).
            :return: Prediction tensor of shape (batch_size, output_window_size, output_vars).
        """
        if ex is not None:
            out_ex = self.mlp(ex[:, -self.ex_retain_window_size:].clone())  # -> (batch_size, input_window_size, hidden_size)

        out_x = self.mlp(x.clone())  # -> (batch_size, ex_retain_window_size, hidden_size)

        if ex is not None:
            out_x = torch.cat([out_ex * self.weight, out_x], 1)  # -> (batch_size, ex_retain_window_size + input_window_size, hidden_size)
        out_x = self.ed(out_x)  # -> (batch_size, ex_retain_window_size, hidden_size)

        out = out_x.clone()
        out0 = self.dr0(out)
        out1 = self.dr1(out)
        out2 = self.dr2(out)
        out = torch.cat([out0, out1, out2], -1)  # -> (batch_size, ex_retain_window_size, 3 * hidden_size)
        out = self.drlin(out) * self.dr_ratio + self.endxlin(out_x) * self.origin_ratio  # -> (batch_size, ex_retain_window_size, output_vars)
        out = self.gar(out)  # -> (batch_size, output_window_size, output_vars)

        return out
