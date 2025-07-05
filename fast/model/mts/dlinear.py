#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from typing import Literal
from ..base.decomposition import DecomposeSeries
from .ar import GAR, AR


class DLinear(nn.Module):
    """
        **Decomposition-Linear**

        Are Transformers Effective for Time Series Forecasting?
        Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu.
        AAAI 2022, DOI: 10.1609/aaai.v37i9.26317.
        url: https://arxiv.org/pdf/2205.13504.pdf

        Author provided code: https://github.com/cure-lab/LTSF-Linear

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param kernel_size: the kernel size of series decomposition function.
        :param mapping: the mapping type, 'gar' for Global AR, 'ar' for Autoregressive.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 kernel_size: int = 25, mapping: Literal['gar', 'ar'] = 'gar'):
        super(DLinear, self).__init__()
        assert mapping in ['gar', 'ar'], f"Mapping should be 'gar' or 'ar', but got {mapping}."
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.kernel_size = kernel_size
        self.mapping = mapping

        self.decomposition = DecomposeSeries(kernel_size)

        if mapping == 'ar':
            self.trend_l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
            self.seasonal_l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
        else:
            self.trend_l1 = GAR(self.input_window_size, self.output_window_size)
            self.seasonal_l1 = GAR(self.input_window_size, self.output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
        """
        trend_init, seasonal_init = self.decomposition(x)

        trend_output = self.trend_l1(trend_init)
        seasonal_output = self.seasonal_l1(seasonal_init)

        out = seasonal_output + trend_output

        return out


class NLinear(nn.Module):
    """
        **Normalization-Linear**

        Are Transformers Effective for Time Series Forecasting?
        Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
        url: https://arxiv.org/pdf/2205.13504.pdf

        Author provided code: https://github.com/cure-lab/LTSF-Linear

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param mapping: the mapping type, 'gar' for Global AR, 'ar' for Autoregressive.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 mapping: Literal['gar', 'ar'] = 'gar'):
        super(NLinear, self).__init__()
        assert mapping in ['gar', 'ar'], f"Mapping should be 'gar' or 'ar', but got {mapping}."
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.mapping = mapping

        if self.mapping == 'ar':
            self.l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
        else:
            self.l1 = GAR(self.input_window_size, self.output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
        """
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        x = self.l1(x)    # -> (batch_size, output_window_size, input_vars)

        out = x + seq_last

        return out
