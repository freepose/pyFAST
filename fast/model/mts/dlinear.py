#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from ..base.decomposition import DecomposeSeries


class DLinear(nn.Module):
    """
        Decomposition-Linear
        Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
        Are Transformers Effective for Time Series Forecasting?
        AAAI 2022, DOI: 10.1609/aaai.v37i9.26317
        url: https://arxiv.org/pdf/2205.13504.pdf

        Official Code: https://github.com/cure-lab/LTSF-Linear

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param individual: whether shared model among different variates.
        :param kernel_size: the kernel size of series decomposition function。
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 individual: bool = False, kernel_size: int = 25):
        super(DLinear, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.individual = individual

        self.decomposition = DecomposeSeries(kernel_size)

        if self.individual:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            for i in range(self.input_vars):
                self.linear_seasonal.append(nn.Linear(self.input_window_size, self.output_window_size))
                self.linear_trend.append(nn.Linear(self.input_window_size, self.output_window_size))
        else:
            self.linear_seasonal = nn.Linear(self.input_window_size, self.output_window_size)
            self.linear_trend = nn.Linear(self.input_window_size, self.output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape is (batch_size, input_window_size, input_vars).
        """
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.transpose(1, 2), trend_init.transpose(1, 2)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.output_window_size],
                                          dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.output_window_size],
                                       dtype=trend_init.dtype, device=trend_init.device)
            for i in range(self.input_vars):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        out = seasonal_output + trend_output
        out = out.transpose(1, 2)

        return out


class NLinear(nn.Module):
    """
        Normalization-Linear
        Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
        Are Transformers Effective for Time Series Forecasting?
        url: https://arxiv.org/pdf/2205.13504.pdf

        Official Code: https://github.com/cure-lab/LTSF-Linear

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param individual: whether shared model among different variates.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 individual: bool = False):
        super(NLinear, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.individual = individual

        if self.individual:
            self.linear = nn.ModuleList()
            for i in range(self.input_vars):
                self.linear.append(nn.Linear(self.input_window_size, self.output_window_size))
        else:
            self.linear = nn.Linear(self.input_window_size, self.output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape is (batch_size, input_window_size, input_vars).
        """
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.output_window_size, x.size(2)], dtype=x.dtype, device=x.device)
            for i in range(self.input_vars):
                output[:, :, i] = self.linear[i](x[:, :, i])
            x = output
        else:
            x = self.linear(x.transpose(1, 2)).transpose(1, 2)
        out = x + seq_last

        return out
