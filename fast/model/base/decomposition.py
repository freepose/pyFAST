#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _scalar_or_tuple_any_t


class MovingAverage(nn.Module):
    """
        Moving Average Block is used to smooth out time series data by averaging values over a specified period.
        (1) Noise Reduction: It helps to reduce the noise in data, making patterns more discernible.
        (2) Trend Analysis: Smoothing out fluctuations helps in identifying trends over time.
        (3) Data Preparation: Preparing features for machine learning models,
            making them more stable and less volatile.
        :param kernel_size: the kernel size of moving average, which is an odd value.
        :param stride: the distance between two adjacent windows.
    """
    def __init__(self, kernel_size: int = 1, stride: int = 1):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad_size = (self.kernel_size - 1) // 2

        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: [batch_size, seq_len, num_features]
        """
        # padding on the both ends of time series
        x = x.permute(0, 2, 1)  # -> [batch_size, num_features, seq_len]
        x = nn.functional.pad(x, (self.pad_size, self.pad_size), mode='replicate')
        x = self.avg(x)  # -> [batch_size, num_features, seq_len // stride]
        x = x.permute(0, 2, 1)  # -> [batch_size, seq_len // stride, num_features]
        return x


class DecomposeSeries(nn.Module):
    """
        Decompose a series into moving average part and residual part.
    """

    def __init__(self, kernel_size: int = 1):
        """
            :param kernel_size: the kernel size of moving average, which is an odd value.
        """
        super(DecomposeSeries, self).__init__()
        self.kernel_size = kernel_size
        self.ma = MovingAverage(self.kernel_size, 1)  # stride is fixed at 1

    def forward(self, x: torch.Tensor) -> _scalar_or_tuple_any_t[torch.Tensor]:
        """
            :param x: [batch_size, seq_len, num_features]
        """
        moving_mean = self.ma(x)  # -> [batch_size, seq_len, num_features]
        residual = x - moving_mean  # -> [batch_size, seq_len, num_features]

        return moving_mean, residual


class DecomposeSeriesMultiKernels(nn.Module):
    """
        Decompose a series into moving average part and residual part. (Fedformer)
    """

    def __init__(self, *kernel_sizes: _size_1_t):
        """
            :param kernel_sizes: the kernel size of moving average, which is a set of odd values.
                                 the default value is 1.
        """
        super(DecomposeSeriesMultiKernels, self).__init__()
        assert len(kernel_sizes) > 1
        self.kernel_sizes = kernel_sizes
        self.kernel_num = len(kernel_sizes)

        self.mas = [MovingAverage(k, 1) for k in self.kernel_sizes]
        self.l1 = nn.Linear(1, self.kernel_num)

    def forward(self, x: torch.Tensor) -> _scalar_or_tuple_any_t[torch.Tensor]:
        """
            :param x: [batch_size, seq_len, num_features]
        """
        moving_mean_list = list(map(lambda ma: ma(x).unsqueeze(-1), self.mas))
        moving_mean_tensor = torch.cat(moving_mean_list, dim=-1)

        x_out = self.l1(x.unsqueeze(-1)).softmax(-1)  # -> [batch_size, seq_len, num_features, kernel_num]
        moving_mean = (moving_mean_tensor * x_out).sum(dim=-1)  # -> [batch_size, seq_len, num_features]

        residual = x - moving_mean  # -> [batch_size, seq_len, num_features]
        return moving_mean, residual
