#!/usr/bin/env python
# encoding: utf-8

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int,
                 dropout_rate: float):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
        Also known as TCN.

        Shaojie Bai and J. Zico Kolter and Vladlen Koltun
        An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
        url: https://arxiv.org/abs/1803.01271

        Official Code: https://github.com/locuslab/TCN

        Assure that ``input_vars`` == ``output_vars``. The convolutional layers are causal on **time dimension**.

        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param num_channels: the number of channels per convolutional layers.
        :param kernel_size: the kernel size of convolutional layers.
        :param dropout_rate: dropout rate.
     """

    def __init__(self, input_window_size: int = 1, output_window_size: int = 1,
                 num_channels: List[int] = [16], kernel_size: int = 2, dropout_rate: float = 0.2):
        super(TemporalConvNet, self).__init__()

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        num_channels.append(output_window_size)
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = input_window_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size,
                                     (kernel_size - 1) * dilation_size, dropout_rate)]

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.network(x)
