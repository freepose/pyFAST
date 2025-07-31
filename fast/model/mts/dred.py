#!/usr/bin/env python
# encoding: utf-8

from typing import Literal

import torch
import torch.nn as nn

from ..base import DirectionalRepresentation
from .ar import GAR


class DRED(nn.Module):
    """
        Directional Representation Encoder-Decoder for Personalized Blood Glucose Forecasting.
        Yu Chen, Zhijin Wang, Jinmo Tang, Henghong Lin, Senzhen Wu, Yaohui Huang.
        PAKDD 2025.
        url: https://link.springer.com/chapter/10.1007/978-981-96-8197-6_15

        Author: Senzhen Wu

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param rnn_cls: type of RNN model ('rnn', 'lstm', 'gru').
        :param hidden_size: number of features in the hidden state.
        :param num_layers:  number of recurrent layers.
        :param bidirectional: whether to use bidirectional RNN or not.
        :param dropout_rate: dropout rate for regularization.
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int = 1, output_vars: int = 1,
                 rnn_cls: Literal['rnn', 'gru', 'lstm'] = 'lstm', hidden_size: int = 512, num_layers: int = 2,
                 bidirectional: bool = True, dropout_rate: float = 0.):
        super(DRED, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate if num_layers > 1 else 0.

        rnn_cls_dict = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        model_cls = rnn_cls_dict[rnn_cls]

        self.rnn_encoder = model_cls(input_vars, hidden_size, num_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=dropout_rate)
        self.rnn_decoder = model_cls(input_vars, hidden_size, num_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=dropout_rate)
        self.l1 = nn.Linear(hidden_size * (2 if bidirectional else 1), output_vars)

        self.dr0 = DirectionalRepresentation(input_window_size, input_vars, r_dim=0, dropout_rate=self.dropout_rate)
        self.dr1 = DirectionalRepresentation(input_window_size, input_vars, r_dim=1, dropout_rate=self.dropout_rate)

        self.gar = GAR(input_window_size, output_window_size)
        self.gar2 = GAR(input_window_size, output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars)
            :return: shape is (batch_size, output_window_size, output_vars)
        """

        x0 = self.dr0(x)
        x1 = self.dr1(x)
        x = x + x0 + x1

        # x -> (batch_size, input_window_size, input_vars)
        _, encoder_hidden = self.rnn_encoder(x)
        # _ -> (batch_size, input_window_size, hidden_size * (2 if bidirectional else 1))
        # encoder_hidden -> 2 * (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)

        decoder_hidden = encoder_hidden

        decoder_output, decoder_hidden = self.rnn_decoder(x, decoder_hidden)
        # decoder_output -> (batch_size, input_window_size, hidden_size * (2 if bidirectional else 1))
        # encoder_hidden -> 2 * (num_layers * (2 if bidirectional else 1), batch_size, hidden_size)

        out = self.l1(decoder_output)  # -> (batch_size, input_window_size, output_vars)
        outputs = self.gar(out)  # -> (batch_size, output_window_size, output_vars)
        outputs = outputs + self.gar2(x)

        return outputs
