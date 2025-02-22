#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from .ar import GAR


class LSTNet(nn.Module):
    """
        Guokun Lai, Wei-Cheng Chang, Yiming Yang, Hanxiao Liu
        Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks
        SIGIR 2018. url: https://arxiv.org/abs/1703.07015

        LSTNet has two major shortcomings:
        (1) the skip length of the recurrent-skip layer must be manually tuned,
            whereas the proposed approach learns the periodic patterns by itself;
        (2) the LSTNet model is specially designed for MTS dataset with strong periodic patterns,
            whereas the proposed attention mechanism, as shown in our experiments,
            is simple and adaptable to various datasets,
            even non-periodic and non-linear ones.
        (3) NOTE: one-step-ahead prediction, by Zhijin Wang.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param cnn_out_channels: the output channel number of CNN part.
        :param cnn_kernel_size: the kernel size of CNN part.
        :param rnn_hidden_size: the hidden size of RNN part.
        :param rnn_num_layers: the layers number of RNN layers.
        :param skip_window_size: the skip window size.
        :param skip_gru_hidden_size: the hidden size of skip GRU.
        :param highway_window_size: the highway window size.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_window_size, input_vars, output_window_size=1, output_vars=1,
                 cnn_out_channels: int = 50, cnn_kernel_size: int = 9,  # CNN parameters
                 rnn_hidden_size: int = 50, rnn_num_layers: int = 1,  # GRU parameters
                 skip_window_size: int = 24, skip_gru_hidden_size: int = 20,  # skip GRU parameters
                 highway_window_size: int = 24, dropout_rate: float = 0.):
        """ skip_window_size is used to reshape time series. """
        super(LSTNet, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.cnn_out_channels = cnn_out_channels
        self.cnn_kernel_size = cnn_kernel_size

        self.rnn_input_size = self.cnn_out_channels
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        self.skip_window_size = skip_window_size  # can be divided by window_size
        self.skip_gru_hidden_size = skip_gru_hidden_size

        self.highway_window_size = highway_window_size

        self.cnn = nn.Conv1d(self.input_vars, self.cnn_out_channels,
                             self.cnn_kernel_size, padding=self.cnn_kernel_size // 2)

        self.rnn = nn.GRU(self.rnn_input_size, self.rnn_hidden_size, batch_first=True)

        if self.skip_window_size > 0:
            self.skip_gru = nn.GRU(self.cnn_out_channels, self.skip_gru_hidden_size, batch_first=True)
            self.linear1 = nn.Linear(self.rnn_hidden_size + self.skip_window_size * self.skip_gru_hidden_size,
                                     self.output_window_size * self.output_vars)
        else:
            self.linear1 = nn.Linear(self.rnn_hidden_size, self.output_window_size * self.output_vars)

        if self.highway_window_size > 0:
            self.highway = GAR(self.highway_window_size * self.input_vars,
                               self.output_window_size * self.output_vars)

        self.d1 = nn.Dropout(dropout_rate)
        self.d2 = nn.Dropout(dropout_rate)
        self.d3 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size, input_size] """

        # CNN
        c = x.permute(0, 2, 1)  # -> [batch_size, input_size, input_window_size]
        c = self.cnn(c)  # -> [batch_size, cnn_out_channels, input_window_size]
        c = torch.relu(c)
        c = self.d1(c)

        # RNN
        r = c.permute(0, 2, 1)  # -> [batch_size, input_window_size, cnn_out_channels]
        _, r = self.rnn(r)  # -> [rnn_num_layers, batch_size, rnn_hidden_size]
        r = r.squeeze(0)  # -> [batch_size, rnn_hidden_size]
        r = self.d2(r)

        # skip-RNN
        if self.skip_window_size > 0:
            # -> [batch_size, out_channels, periodic_size, skip_window_size]
            s = c.reshape(c.shape[0], c.shape[1], -1, self.skip_window_size)
            s = s.permute(0, 3, 2, 1)  # -> [batch_size, skip_window_size, periodic_size, out_channels]
            s = s.reshape(-1, s.shape[2], s.shape[3])  # -> [batch_size * skip_window_size, periodic_size, out_channels]
            _, s = self.skip_gru(s)  # -> [num_layers, batch_size * skip_window_size, skip_gru_hidden_size]
            # -> [batch_size, skip_window_size * skip_gru_hidden_size]
            s = s.reshape(-1, self.skip_window_size * self.skip_gru_hidden_size)
            r = torch.cat([r, s], 1)  # -> [batch_size, rnn_hidden_size + skip_window_size * skip_gru_hidden_size]

        res = self.linear1(r)  # -> [batch_size, output_window_size * output_size]
        res = res.reshape(-1, self.output_window_size,
                          self.output_vars)  # -> [batch_size, output_window_size, output_size]

        # highway
        if self.highway_window_size > 0:
            z = x[:, -self.highway_window_size:, :]  # -> [batch_size, highway_window_size, input_size]
            z = z.reshape(-1, self.highway_window_size * self.input_vars,
                          1)  # -> [batch_size, highway_window_size * input_size, 1]
            z = self.highway(z)  # -> [batch_size, output_window_size * output_size, 1]
            z = z.reshape(-1, self.output_window_size,
                          self.output_vars)  # -> [batch_size, output_window_size, output_size]
            res = res + z  # -> [batch_size, output_window_size, output_size]

        return res