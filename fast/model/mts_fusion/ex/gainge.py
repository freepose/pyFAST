#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from ....model.mts import GAR
from ....model.mts.gnn.gain import GAT


class GAINGE(nn.Module):
    """
        Wang Z, Liu X, Huang Y, et al.
        A multivariate time series graph neural network for district heat load forecasting[J].
        Energy, 2023, 278: 127911.
        URL: https://doi.org/10.1016/j.energy.2023.127911

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param ex_vars: number of exogenous input variables.
        :param output_window_size: output window size.
        :param gat_h_dim: the hidden dimension of GAT.
        :param dropout_rate: dropout rate.
        :param highway_window_size: the length of highway window size.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, ex_vars: int = 1, output_window_size: int = 1,
                 gru_hidden_size: int = 8, cnn_kernel_size: int = 3, cnn_out_channels: int = 16,
                 ex_cnn_out_channels: int = 10,
                 gat_hidden_size: int = 64, gat_h_dim: int = 4, ex_gat_hidden_size: int = 4, ex_gat_h_dim: int = 4,
                 gate_hidden_size: int = 16, gate_h_dim: int = 4,
                 dropout_rate: float = 0.01, highway_window_size: int = 10):
        super(GAINGE, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size

        self.gru_hidden_size = gru_hidden_size
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_out_channels = cnn_out_channels
        self.ex_cnn_out_channels = ex_cnn_out_channels

        self.gat_hidden_size = gat_hidden_size
        self.gat_h_dim = gat_h_dim
        self.ex_gat_hidden_size = ex_gat_hidden_size
        self.ex_gat_h_dim = ex_gat_h_dim
        self.gate_hidden_size = gate_hidden_size
        self.gate_h_dim = gate_h_dim

        self.dropout_rate = dropout_rate
        self.highway_window_size = highway_window_size

        self.gat1 = GAT(self.cnn_out_channels, self.gat_hidden_size, n_head=self.gat_h_dim,
                        dropout_rate=self.dropout_rate, last=False)
        self.gate1 = GAT(self.ex_vars, self.gate_hidden_size, n_head=self.gate_h_dim, dropout_rate=self.dropout_rate,
                         last=False)

        self.conv1 = nn.Conv1d(self.input_vars,
                               self.cnn_out_channels, self.cnn_kernel_size, padding=self.cnn_kernel_size // 2)

        self.conve = nn.Conv1d(self.ex_vars, self.ex_cnn_out_channels,
                               self.cnn_kernel_size, padding=self.cnn_kernel_size // 2)

        self.highway = GAR(self.highway_window_size, self.output_window_size)
        self.ex_highway = GAR(self.input_window_size, self.output_window_size)

        self.gru1 = nn.GRU(
            self.gate_hidden_size * self.gate_h_dim + 1 * self.gat_hidden_size * self.gat_h_dim + self.input_vars,
            self.gru_hidden_size, batch_first=True)

        self.out = nn.Linear(1 * self.gru_hidden_size, self.input_vars)
        self.eo = nn.Linear(self.ex_cnn_out_channels, self.input_vars)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: (batch_size, input_window_size, input_vars)
            :param ex: (batch_size, input_window_size, ex_vars)
            :return: (batch_size, output_window_size, input_vars)
        """

        x1 = x
        e = ex

        e = e.permute(0, 2, 1)
        e = self.conve(e)
        e = e.permute(0, 2, 1)

        e = self.ex_highway(e)
        e = self.eo(e)

        r1 = x1.permute(0, 2, 1)
        r1 = self.conv1(r1)
        r1 = r1.permute(0, 2, 1)
        r1 = self.gat1(r1)
        y1 = self.gate1(ex)
        _, r1 = self.gru1(torch.cat([r1, x1, y1], dim=2))
        r1 = r1.squeeze(0)
        r1 = r1.reshape(-1, 1, self.gru_hidden_size)

        r = self.out(r1)

        if self.highway_window_size > 0:
            z = x[:, -self.highway_window_size:, :]  # => [batch_size, highway_window_size, input_size]
            z = self.highway(z)  # => [batch_size, 1, input_size]
            r = r + z + e

        return r
