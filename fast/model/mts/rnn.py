#!/usr/bin/env python
# encoding: utf-8

"""
    The model of rnn series: RNN, GRU, LSTM; BiRNN series, BiLSTM.
"""

import torch
import torch.nn as nn

from typing import Literal
from ..base import MinLSTM


class TimeSeriesRNN(nn.Module):
    """
        Recurrent Neural Network for time series forecasting. The input window is variable length.

        Decoder-only RNN series.

        Ian Fox, Lynn Ang, Mamta Jaiswal, Rodica Pop-Busui, Jenna Wiens.
        Deep Multi-Output Forecasting: Learning to Accurately Predict Blood Glucose Trajectories. KDD 2018.
        url: https://arxiv.org/abs/1806.05357

        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param output_vars: output variable(s) number.
        :param rnn_cls: the rnn type is in ['rnn', 'lstm', 'gru', 'minlstm'].
        :param hidden_size: hidden size of rnn.
        :param num_layers: number of rnn layers.
        :param bidirectional: whether to use bidirectional rnn or not.
        :param dropout_rate: dropout rate.
        :param decoder_way: the decoder way is in ['inference', 'mapping']. In KDD 2018 Glucose,
                            the 'inference' is also called 'recursive'. The 'mapping' is also called 'multi-output'.
    """

    def __init__(self, input_vars: int, output_window_size: int = 1, output_vars: int = 1,
                 rnn_cls: Literal['rnn', 'lstm', 'gru', 'minlstm'] = 'gru',
                 hidden_size: int = 32, num_layers: int = 1, bidirectional: bool = False,
                 dropout_rate: float = 0., decoder_way: Literal['inference', 'mapping'] = 'inference'):
        assert rnn_cls in ['rnn', 'lstm', 'gru', 'minlstm'], "rnn_cls must be 'rnn', 'lstm', 'gru', 'minlstm'"
        assert decoder_way in ['inference', 'mapping'], "decoder_way must be 'inference' or 'mapping'"

        super(TimeSeriesRNN, self).__init__()
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.rnn_cls = rnn_cls
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.decoder_way = decoder_way
        self.dropout_rate = dropout_rate if self.num_layers > 1 else 0.

        rnn_cls_dict = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'minlstm': MinLSTM}

        self.l0 = nn.Linear(self.input_vars, self.input_vars, bias=False)

        model_cls = rnn_cls_dict.get(self.rnn_cls)
        self.rnn = model_cls(self.input_vars, self.hidden_size, self.num_layers, batch_first=True,
                             bidirectional=self.bidirectional, dropout=self.dropout_rate)

        rnn_out_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        if self.decoder_way == 'inference':
            self.l1 = nn.Linear(rnn_out_dim, self.input_vars)
            if self.input_vars != self.output_vars:
                self.l2 = nn.Linear(self.input_vars, self.output_vars)
        else:
            self.l3 = nn.Linear(rnn_out_dim, self.output_vars)

    def forward(self, x):
        """
        :param x: shape is (batch_size, input_window_size, input_vars).
        """
        x = self.l0(x)  # -> (batch_size, input_window_size, input_vars)
        x = x.relu()

        outputs = torch.zeros(x.shape[0], self.output_window_size, self.output_vars, dtype=x.dtype, device=x.device)

        _, hidden = self.rnn(x)

        if self.decoder_way == 'inference':
            # Decoder: inference
            inputs = x[:, -1:, :]
            for t in range(self.output_window_size):
                rnn_output, hidden = self.rnn(inputs, hidden)   # -> (batch_size, 1, hidden_size)
                out = self.l1(rnn_output)   # -> (batch_size, 1, input_vars)
                inputs = out
                outputs[:, t:t+1, :] = self.l2(out) if self.l2 is not None else out
        else:
            # Decoder: mapping, assure that input_window_size >= output_window_size
            rnn_output, hidden = self.rnn(x, hidden)    # -> (batch_size, input_window_size, hidden_size)
            out = self.l3(rnn_output)   # -> (batch_size, input_window_size, output_vars)
            outputs = out[:, -self.output_window_size:, :]

        return outputs


class EncoderDecoder(nn.Module):
    """
        Encoder-decoder framework for time series forecasting.

        Encoder-Decoder RNN series.

        :param input_vars: Number of input features.
        :param output_window_size: Number of time steps to predict.
        :param output_vars: Number of output features.
        :param rnn_cls: Type of RNN model ('rnn', 'lstm', 'gru', 'minlstm').
        :param hidden_size: Number of features in the hidden state.
        :param num_layers: Number of recurrent layers.
        :param bidirectional: Whether to use bidirectional RNN or not.
        :param dropout_rate: Dropout rate for regularization.
        :param decoder_way: the decoder way is in ['inference', 'mapping']. In KDD 2018,
                            the 'inference' is also called 'recursive'. The 'mapping' is also called 'multi-output'.
    """

    def __init__(self, input_vars: int, output_window_size: int = 1, output_vars: int = 1,
                 rnn_cls: Literal['rnn', 'lstm', 'gru', 'minlstm'] = 'gru', hidden_size: int = 10,
                 num_layers: int = 2, bidirectional: bool = False, dropout_rate: float = 0.,
                 decoder_way: Literal['inference', 'mapping'] = 'inference'):
        assert rnn_cls in ['rnn', 'lstm', 'gru', 'minlstm'], "rnn_cls must be 'rnn', 'lstm', or 'gru'"
        assert decoder_way in ['inference', 'mapping'], "decoder_way must be 'inference' or 'mapping'"

        super(EncoderDecoder, self).__init__()  # Initialize the nn.Module parent class
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.decoder_way = decoder_way
        self.dropout_rate = dropout_rate if num_layers > 1 else 0.

        assert rnn_cls in ['rnn', 'lstm', 'gru', 'minlstm'], "rnn_cls must be 'rnn', 'lstm', 'gru' or 'minlstm'"

        rnn_cls_dict = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'minlstm': MinLSTM}
        model_cls = rnn_cls_dict[rnn_cls]

        self.rnn_encoder = model_cls(self.input_vars, self.hidden_size, self.num_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=self.dropout_rate)
        self.rnn_decoder = model_cls(self.input_vars, self.hidden_size, self.num_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=self.dropout_rate)

        rnn_out_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        if self.decoder_way == 'inference':
            self.l1 = nn.Linear(rnn_out_dim, self.input_vars)
            if self.input_vars != self.output_vars:
                self.l2 = nn.Linear(self.input_vars, self.output_vars)
        else:
            self.l3 = nn.Linear(rnn_out_dim, self.output_vars)

        # Used to facilitate multi-GPUs (if needed)
        self.module_list = nn.ModuleList([self.rnn_encoder, self.rnn_decoder])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: Input tensor of shape (batch_size, input_window_size, input_vars).
            :return: Prediction tensor of shape (batch_size, output_window_size, output_vars).
        """
        # Encode the input sequence
        encoder_out, encoder_hidden = self.rnn_encoder(x)

        # Initialize tensor for predictions
        outputs = torch.zeros(x.shape[0], self.output_window_size, self.output_vars, dtype=x.dtype, device=x.device)

        decoder_hidden = encoder_hidden

        if self.decoder_way == 'inference':
            # Decoder: inference
            decoder_input = x[:, -1:, :]  # Set initial decoder input as the last value of input sequence
            for t in range(self.output_window_size):
                decoder_output, decoder_hidden = self.rnn_decoder.forward(decoder_input, decoder_hidden)
                out = self.l1(decoder_output)  # -> (batch_size, 1, input_vars)
                decoder_input = out
                outputs[:, t:t + 1, :] = self.l2(out) if self.l2 is not None else out
        else:
            # Decoder: mapping, assure that input_window_size >= output_window_size
            decoder_output, decoder_hidden = self.rnn_decoder(x, decoder_hidden)
            out = self.l3(decoder_output)  # -> (batch_size, input_window_size, output_vars)
            outputs = out[:, -self.output_window_size:, :]

        return outputs
