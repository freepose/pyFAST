#!/usr/bin/env python
# encoding: utf-8

"""
    Multi-view Time Series Model (MvT).
"""
import torch
import torch.nn as nn

from ..base.dr import DirectionalRepresentation
from ..mts import GAR


class MvT(nn.Module):
    """
        Wang, Zhijin, et al.
        A multi-view time series model for share turnover prediction.
        Applied Intelligence (2022): 1-12.
        Author: Zhijin Wang

        Multi-view Time Series Model (MvT).

        :param input_window_size: input window size of target variable and exogenous variables.
        :param input_vars: number of input variables.
        :param output_window_size: output window size, a.k.a., prediction length.
        :param ex_retain_window_size: input window size of exogenous variables.
        :param ex_vars: number of exogenous variables.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = 1, dropout_rate: float = 0.):
        super(MvT, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        # For x
        self.dt_x0 = DirectionalRepresentation(self.input_window_size, self.input_vars, 0, dropout_rate=dropout_rate)
        self.dt_x1 = DirectionalRepresentation(self.input_window_size, self.input_vars, 1, dropout_rate=dropout_rate)
        self.dt_x2 = DirectionalRepresentation(self.input_window_size, self.input_vars, 2, dropout_rate=dropout_rate)
        self.ar_x4 = GAR(self.input_window_size, self.output_window_size)
        self.ar_x = GAR(self.input_window_size, self.output_window_size)

        # For ex
        self.dt_ex0 = DirectionalRepresentation(self.ex_retain_window_size, self.ex_vars, 0, dropout_rate=dropout_rate)
        self.dt_ex1 = DirectionalRepresentation(self.ex_retain_window_size, self.ex_vars, 1, dropout_rate=dropout_rate)
        self.dt_ex2 = DirectionalRepresentation(self.ex_retain_window_size, self.ex_vars, 2, dropout_rate=dropout_rate)
        self.ar_ex4 = GAR(self.ex_retain_window_size, self.output_window_size)
        self.ar_ex = GAR(self.ex_retain_window_size, self.output_window_size)

        # Output
        self.l1 = nn.Linear(self.input_vars * 4 + self.ex_vars * 4, self.input_vars)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: [batch_size, input_window_size, input_vars]
            :param ex: [batch_size, ex_retain_window_size, ex_vars]
        """
        # For x
        out_x0 = self.dt_x0(x)  # => [batch_size, input_window_size, input_vars]
        out_x1 = self.dt_x1(x)  # => [batch_size, input_window_size, input_vars]
        out_x2 = self.dt_x2(x)  # => [batch_size, input_window_size, input_vars]
        out_x4 = self.ar_x4(x)  # => [batch_size, output_window_size, input_vars]

        out_x = torch.cat([out_x0, out_x1, out_x2], dim=2)  # => [batch_size, input_window_size, input_vars * 3]
        out_x = self.ar_x(out_x)  # => [batch_size, output_window_size, input_vars * 3]

        # For ex
        retain_ex = ex[:, -self.ex_retain_window_size:]
        out_ex0 = self.dt_ex0(retain_ex)  # => [batch_size, ex_retain_window_size, ex_vars]
        out_ex1 = self.dt_ex1(retain_ex)  # => [batch_size, ex_retain_window_size, ex_vars]
        out_ex2 = self.dt_ex2(retain_ex)  # => [batch_size, ex_retain_window_size, ex_vars]
        out_ex4 = self.ar_ex4(retain_ex)  # => [batch_size, output_window_size, ex_vars]

        out_ex = torch.cat([out_ex0, out_ex1, out_ex2], dim=2)  # => [batch_size, ex_retain_window_size, ex_vars * 3]
        out_ex = self.ar_ex(out_ex)         # => [batch_size, output_window_size, ex_vars * 3]

        # => [batch_size, output_window_size, ex_vars * 4 + input_vars * 4]
        out = torch.cat([out_ex, out_x, out_ex4, out_x4], dim=2)

        out = self.l1(out)
        out = self.dropout(out)

        return out

