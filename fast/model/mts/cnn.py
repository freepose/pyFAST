#!/usr/bin/env python
# encoding: utf-8

from typing import List, Literal

import torch
import torch.nn as nn

from .rnn import TimeSeriesRNN


class CNNRNN(nn.Module):
    """
        Yuexin Wu, Yiming Yang, Hiroshi Nishiura, Masaya Saitoh
        Deep Learning for Epidemiological Predictions
        SIGIR 2018, pp. 1085 - 1088
        url: https://dl.acm.org/doi/10.1145/3209978.3210077

        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param output_vars: output size.
        :param cnn_out_channels: number of cnn out channels.
        :param cnn_kernel_size: kernel size of cnn.
        :param rnn_cls: rnn, lstm, gru.
        :param rnn_hidden_size: hidden size of rnn.
        :param rnn_num_layers: number of rnn layers.
        :param rnn_bidirectional: whether to use bidirectional rnn or not.
        :param dropout_rate: dropout rate.
        :param decoder_way: the decoder way is in ['inference', 'mapping']. In KDD 2018,
                            the 'inference' is also called 'recursive'. The 'mapping' is also called 'multi-output'.
    """

    def __init__(self, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 cnn_out_channels: int = 50, cnn_kernel_size: int = 9,
                 rnn_cls: Literal['rnn', 'lstm', 'gru'] = 'gru',
                 rnn_hidden_size: int = 20, rnn_num_layers: int = 1, rnn_bidirectional: bool = False,
                 dropout_rate: float = 0., decoder_way: Literal['inference', 'mapping'] = 'inference'):
        super(CNNRNN, self).__init__()
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.cnn_out_channels = cnn_out_channels
        self.cnn_kernel_size = cnn_kernel_size

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional

        self.conv1d = nn.Conv1d(in_channels=self.input_vars, out_channels=self.cnn_out_channels,
                                kernel_size=self.cnn_kernel_size, padding=self.cnn_kernel_size // 2)

        self.gru = TimeSeriesRNN(self.cnn_out_channels, self.output_window_size, self.output_vars, rnn_cls,
                                 self.rnn_hidden_size, self.rnn_num_layers, self.rnn_bidirectional,
                                 dropout_rate, decoder_way)

        self.d1 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, window_size, input_size] """
        x = x.permute(0, 2, 1)  # => [batch_size, input_size, input_window_size]
        x = self.conv1d(x)  # => [batch_size, cnn_out_channels, input_window_size]
        x = torch.relu(x)
        x = self.d1(x)

        x = x.permute(0, 2, 1)  # => [batch_size, input_window_size, cnn_out_channels]
        out = self.gru(x)
        return out


class CNNRNNRes(nn.Module):
    """
        Yuexin Wu, Yiming Yang, Hiroshi Nishiura, Masaya Saitoh
        Deep Learning for Epidemiological Predictions
        SIGIR 2018, pp. 1085 - 1088
        url: https://dl.acm.org/doi/10.1145/3209978.3210077

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param output_vars: output size.
        :param cnn_out_channels: number of cnn out channels.
        :param cnn_kernel_size: kernel size of cnn.
        :param rnn_cls: rnn, lstm, gru.
        :param rnn_hidden_size: hidden size of rnn.
        :param rnn_num_layers: number of rnn layers.
        :param rnn_bidirectional: whether to use bidirectional rnn or not.
        :param dropout_rate: dropout rate.
        :param decoder_way: the decoder way is in ['inference', 'mapping']. In KDD 2018,
                            the 'inference' is also called 'recursive'. The 'mapping' is also called 'multi-output'.
        :param residual_window_size: residual window size.
        :param residual_ratio: the residual ratio of outputs.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 cnn_out_channels: int = 50, cnn_kernel_size: int = 9,
                 rnn_cls: Literal['rnn', 'lstm', 'gru'] = 'gru',
                 rnn_hidden_size: int = 20, rnn_num_layers: int = 1, rnn_bidirectional: bool = False,
                 dropout_rate: float = 0., decoder_way: Literal['inference', 'mapping'] = 'inference',
                 residual_window_size: int = 5, residual_ratio: float = 0.1, ):
        assert input_window_size >= residual_window_size, 'residual_window_size must be smaller equal to input_window_size'

        super(CNNRNNRes, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.cnn_out_channels = cnn_out_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_padding = self.cnn_kernel_size // 2

        self.rnn_input_size = self.cnn_out_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional

        self.residual_window_size = residual_window_size
        self.residual_ratio = residual_ratio

        self.cnn = nn.Conv1d(in_channels=self.input_vars, out_channels=self.cnn_out_channels,
                             kernel_size=self.cnn_kernel_size, padding=self.cnn_padding)
        self.dropout = nn.Dropout(dropout_rate)

        self.rnn = TimeSeriesRNN(self.cnn_out_channels, self.output_window_size, self.output_vars, rnn_cls,
                                 self.rnn_hidden_size, self.rnn_num_layers, self.rnn_bidirectional,
                                 dropout_rate, decoder_way)

        if self.residual_window_size > 0:
            self.residual_window_size = self.residual_window_size
            self.residual = nn.Linear(self.residual_window_size * self.input_vars,
                                      self.output_window_size * self.output_vars)

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size, input_size] """
        res = x.permute(0, 2, 1)  # => [batch_size, input_size, input_window_size]
        res = self.cnn(res)  # => [batch_size, cnn_out_channels, input_window_size]
        res = torch.relu(res)
        res = self.dropout(res)

        res = res.permute(0, 2, 1)  # => [batch_size, input_window_size, cnn_out_channels]
        res = self.rnn(res)  # => [batch_size, output_window_size, output_size]

        if self.residual_window_size > 0:
            z = x[:, -self.residual_window_size:, :]  # => [batch_size, residual_window_size, input_size]
            z = z.permute(0, 2, 1)  # => [batch_size, input_size, residual_window_size]
            z = z.reshape(-1, self.residual_window_size * self.input_vars)  # => [batch_size, ...]
            z = self.residual(z)  # => [batch_size, output_size * output_window_size]
            z = z.view(-1, self.output_window_size, self.output_vars)  # => [batch_size, ...]
            res = res * self.residual_ratio + z

        return res
