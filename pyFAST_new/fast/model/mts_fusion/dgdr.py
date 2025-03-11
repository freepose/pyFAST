#!/usr/bin/env python
# encoding: utf-8

"""
    Dual-grained Directional Representation (DGDR).
"""
import torch
import torch.nn as nn

from ..base.dr import DirectionalRepresentation
from ..mts import GAR


class DGDR(nn.Module):
    """
        Zhang, Peisong, et al.
        Dual-grained directional representation for infectious disease case prediction.
        Knowledge-Based Systems 256 (2022): 109806.

        Author: Zhijin Wang

        :param input_window_size: input window size of target variable and exogenous variables.
        :param input_vars: number of input variables.
        :param output_window_size: output window size, a.k.a., prediction length.
        :param ex_vars: number of exogenous variables.
        :param dropout_rate: dropout rate.
    """
    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1,
                 ex_vars: int = 1, dropout_rate: float = 0.):
        """ Note that, input_window_size may be not equal to exogenous_input_window_size. """
        super(DGDR, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.ex_vars = ex_vars

        self.ex_t0 = DirectionalRepresentation(self.input_window_size, self.ex_vars, 0, dropout_rate=dropout_rate)
        self.ex_t1 = DirectionalRepresentation(self.input_window_size, self.ex_vars, 1, dropout_rate=dropout_rate)
        self.ex_t2 = DirectionalRepresentation(self.input_window_size, self.ex_vars, 2, dropout_rate=dropout_rate)

        self.target_t0 = DirectionalRepresentation(self.input_window_size, self.input_vars, 0, dropout_rate=dropout_rate)
        self.target_t1 = DirectionalRepresentation(self.input_window_size, self.input_vars, 1, dropout_rate=dropout_rate)
        self.target_t2 = DirectionalRepresentation(self.input_window_size, self.input_vars, 2, dropout_rate=dropout_rate)

        # Regression for all directional representations.
        representation_dim = self.ex_vars * 4 + self.input_vars * 4

        # self.ar = AR(self.input_window_size, representation_dim, self.output_window_size)
        self.ar = GAR(self.input_window_size, self.output_window_size)

        self.l1 = nn.Linear(representation_dim, self.input_vars)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            :param x: slicing window of the target variable. [batch_size, input_window_size, input_vars]
            :param ex: slicing window of exogenous variables. [batch_size, input_window_size, ex_vars]
        """
        target_out0 = self.target_t0(x)     # => [batch_size, input_window_size, input_vars]
        target_out1 = self.target_t1(x)     # => [batch_size, input_window_size, input_vars]
        target_out2 = self.target_t2(x)     # => [batch_size, input_window_size, input_vars]

        ex_out0 = self.ex_t0(ex)     # => [batch_size, input_window_size, ex_input_size]
        ex_out1 = self.ex_t1(ex)     # => [batch_size, input_window_size, ex_input_size]
        ex_out2 = self.ex_t2(ex)     # => [batch_size, input_window_size, ex_input_size]

        out = torch.cat([ex_out0, ex_out1, ex_out2, ex, target_out0, target_out1, target_out2, x], dim=2)
        out = self.ar(out)
        out = self.l1(out)

        return out
