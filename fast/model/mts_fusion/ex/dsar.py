#!/usr/bin/env python
# encoding: utf-8

"""
    Dual Sides Auto-Regression (DSAR).
"""

import torch
import torch.nn as nn

from ...mts import GAR, AR


class DSAR(nn.Module):
    """
        Wang Z, Huang Y, Cai B, et al.
        Stock turnover prediction using search engine data[J].
        Journal of Circuits, Systems and Computers, 2021, 30(07): 2150122.

        Dual Sides Auto-Regression (DSAR).

        :param input_window_size: input window size of target variable and exogenous variables.
        :param input_vars: number of input variables.
        :param output_window_size: output window size, a.k.a., prediction length.
        :param ex_retain_window_size: input window size of exogenous variables.
        :param ex_vars: number of exogenous variables.
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int = 1,
                 ex_retain_window_size: int = None, ex_vars: int = 1):
        super(DSAR, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.ex_retain_window_size = input_window_size if ex_retain_window_size is None else ex_retain_window_size
        self.ex_vars = ex_vars

        gain = 0.01

        self.ex_weight = nn.Parameter(torch.rand(self.ex_retain_window_size, self.ex_vars) * gain)
        self.ex_bias = nn.Parameter(torch.zeros(self.ex_vars))

        self.ar1 = GAR(self.ex_retain_window_size, self.output_window_size)
        # self.ar1 = AR(self.ex_retain_window_size, self.ex_vars, self.output_window_size)
        self.l1 = nn.Linear(self.ex_vars, self.input_vars)

        # parameters of auto-regression on target inputs
        # self.weight = nn.Parameter(torch.rand(self.input_window_size, self.input_vars) * gain)
        # self.bias = nn.Parameter(torch.zeros(self.input_vars))
        self.ar2 = AR(self.input_window_size, self.input_vars, self.output_window_size)

        self.ar3 = GAR(self.output_window_size * 2, self.output_window_size)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: sliding window of the target variable. [batch_size, window_size, input_vars]
            :param ex: sliding window of exogenous variables.
                      [batch_size, ex_retain_window_size, ex_vars],
                   or [batch_size, ex_retain_window_size + output_window_size, ex_vars],
                   or [batch_size, output_window_size, ex_vars],
        """
        out_ex = ex[:, -self.ex_retain_window_size:]
        out_ex = out_ex * self.ex_weight + self.ex_bias     # -> [batch_size, ex_retain_window_size, ex_vars]
        out_ex = out_ex.softmax(dim=1)                  # important: the softmax on the lags

        out_ex = self.ar1(out_ex)     # -> [batch_size, output_window_size, ex_vars]
        out_ex = self.l1(out_ex)      # -> [batch_size, output_window_size, input_vars]

        out_y = self.ar2(x)     # -> [batch_size, output_window_size, input_vars]

        # out = torch.cat([out_x, out_y], dim=1)   # -> [batch_size, output_window_size * 2, input_vars]
        # out = self.ar3(out)

        out = out_ex + out_y

        return out
