#!/usr/bin/env python
# encoding: utf-8

"""
    The model of rnn series: RNN, GRU, LSTM; BiRNN series, BiLSTM.
"""
from typing import Literal

import torch
import torch.nn as nn


class MinLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MinLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)

    def forward(self, x, states):
        h_prev, c_prev = states

        # Concatenate input and hidden state
        combined = torch.cat([x, h_prev], dim=1)

        # Single linear transformation
        gates = self.linear(combined)

        # Split the gates
        i_t, f_t, c_tilde, o_t = gates.chunk(4, dim=1)

        # Apply activations
        i_t = torch.sigmoid(i_t)  # Input gate
        f_t = torch.sigmoid(f_t)  # Forget gate
        c_tilde = torch.tanh(c_tilde)  # Candidate cell state
        o_t = torch.sigmoid(o_t)  # Output gate

        # Update cell state and hidden state
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class MinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0,
                 bidirectional=False):
        """
        A minimal LSTM implementation with support for multiple layers, dropout, and bidirectionality.

        Args:
            input_size (int): The number of features in the input.
            hidden_size (int): The number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            bias (bool): If False, disables the use of bias in the linear layers.
            batch_first (bool): If True, the input and output tensors are provided as (batch, seq, feature).
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of each LSTM layer
                             except the last layer, with dropout probability equal to `dropout`.
            bidirectional (bool): If True, becomes a bidirectional LSTM.
        """
        super(MinLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        num_directions = 2 if bidirectional else 1

        # Define LSTM layers
        self.cells = nn.ModuleList([
            nn.ModuleList([
                MinLSTMCell(input_size if i == 0 else hidden_size * num_directions, hidden_size, bias=bias),
                MinLSTMCell(input_size if i == 0 else hidden_size * num_directions, hidden_size, bias=bias)
            ]) if bidirectional else
            MinLSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias=bias)
            for i in range(num_layers)
        ])

        # Define dropout layers for intermediate layers (except the last one)
        self.dropout_layers = nn.ModuleList([
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
            for _ in range(num_layers - 1)
        ])

    def forward(self, x, states=None):
        """
        Forward pass for the MiniLSTM.

        Args:
            x (Tensor): Input tensor of shape (batch, seq, feature) if batch_first=True,
                        otherwise (seq, batch, feature).
            states (Tuple[Tensor, Tensor]): A tuple containing the initial hidden states and cell states.
                                            Each is of shape (num_layers * num_directions, batch, hidden_size).

        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: The output tensor of shape (batch, seq, hidden_size * num_directions) if batch_first=True,
                                                  otherwise (seq, batch, hidden_size * num_directions),
                                                  and the tuple of the final hidden states and cell states.
        """
        if not self.batch_first:
            x = x.transpose(0, 1)  # Convert to (batch, seq, feature)

        batch_size, seq_len, _ = x.size()
        num_directions = 2 if self.bidirectional else 1

        if states is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in
                 range(self.num_layers * num_directions)]
            c = [torch.zeros(batch_size, self.hidden_size, device=x.device) for _ in
                 range(self.num_layers * num_directions)]
        else:
            h, c = list(states[0]), list(states[1])

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                if self.bidirectional:
                    h_f, c_f = self.cells[layer][0](x_t, (h[layer * num_directions], c[layer * num_directions]))
                    h_b, c_b = self.cells[layer][1](x_t.flip(1),
                                                    (h[layer * num_directions + 1], c[layer * num_directions + 1]))
                    x_t = torch.cat([h_f, h_b], dim=-1)
                    h[layer * num_directions], c[layer * num_directions] = h_f, c_f
                    h[layer * num_directions + 1], c[layer * num_directions + 1] = h_b, c_b
                else:
                    h[layer], c[layer] = self.cells[layer](x_t, (h[layer], c[layer]))
                    x_t = h[layer]

                # Apply dropout if not the last layer
                if layer < self.num_layers - 1:
                    x_t = self.dropout_layers[layer](x_t)

            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq, hidden_size * num_directions)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1)  # Convert back to (seq, batch, feature)

        h = torch.stack(h, dim=0)  # (num_layers * num_directions, batch, hidden_size)
        c = torch.stack(c, dim=0)  # (num_layers * num_directions, batch, hidden_size)

        return outputs, (h, c)


class TimeSeriesRNN(nn.Module):
    """
        Recurrent Neural Network for time series forecasting. The input window is variable length.

        Decoder-only RNN series.

        Ian Fox, Lynn Ang, Mamta Jaiswal, Rodica Pop-Busui, Jenna Wiens.
        Deep Multi-Output Forecasting: Learning to Accurately Predict Blood Glucose Trajectories. KDD 2018.
        url: https://arxiv.org/abs/1806.05357

        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param output_vars: output size.
        :param rnn_cls: rnn, lstm, gru.
        :param hidden_size: hidden size of rnn.
        :param num_layers: number of rnn layers.
        :param bidirectional: whether to use bidirectional rnn or not.
        :param dropout_rate: dropout rate.
        :param decoder_way: the decoder way is in ['inference', 'mapping']. In KDD 2018,
                            the 'inference' is also called 'recursive'. The 'mapping' is also called 'multi-output'.
    """

    def __init__(self, input_vars: int, output_window_size: int = 1, output_vars: int = 1,
                 rnn_cls: Literal['rnn', 'lstm', 'gru', 'minlstm'] = 'gru',
                 hidden_size: int = 32, num_layers: int = 1, bidirectional: bool = False,
                 dropout_rate: float = 0., decoder_way: Literal['inference', 'mapping'] = 'inference'):
        assert input_vars == output_vars, 'input_vars must be equal to output_vars'
        assert rnn_cls in ['rnn', 'lstm', 'gru', 'minlstm'], "rnn_cls must be 'rnn', 'lstm', or 'gru'"
        assert decoder_way in ['inference', 'mapping'], "decoder_way must be 'inference' or 'mapping'"

        super(TimeSeriesRNN, self).__init__()
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.decoder_way = decoder_way
        self.dropout_rate = dropout_rate if num_layers > 1 else 0.

        rnn_cls_dict = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'minlstm': MinLSTM}

        model_cls = rnn_cls_dict.get(rnn_cls)
        self.rnn = model_cls(input_vars, hidden_size, num_layers, batch_first=True,
                             bidirectional=bidirectional, dropout=dropout_rate)

        self.fc1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_vars)

    def forward(self, x):
        """
        :param x: shape is (batch_size, input_window_size, input_vars).
        """
        outputs = torch.zeros(x.shape[0], self.output_window_size, x.shape[2], dtype=x.dtype, device=x.device)

        _, hidden = self.rnn(x)

        if self.decoder_way == 'inference':
            # Decoder: inference
            inputs = x[:, -1:, :]
            for t in range(self.output_window_size):
                rnn_output, hidden = self.rnn(inputs, hidden)
                out = self.fc1(rnn_output)
                outputs[:, t, :] = out.squeeze(1)
                inputs = out
        else:
            # Decoder: mapping, assure that input_window_size >= output_window_size
            rnn_output, hidden = self.rnn(x, hidden)
            out = self.fc1(rnn_output)
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
        assert input_vars == output_vars, 'input_vars must be equal to output_vars'
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

        self.rnn_encoder = model_cls(input_vars, hidden_size, num_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=dropout_rate)
        self.rnn_decoder = model_cls(input_vars, hidden_size, num_layers, batch_first=True,
                                     bidirectional=bidirectional, dropout=dropout_rate)
        self.l1 = nn.Linear(self.hidden_size * 2 if bidirectional else self.hidden_size, self.output_vars)

        # Used to facilitate multi-GPUs (if needed)
        self.module_list = nn.ModuleList([self.rnn_encoder, self.rnn_decoder, self.l1])

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
                out = self.l1(decoder_output)  # -> (batch_size, 1, output_vars)
                outputs[:, t, :] = out.squeeze(1)
                decoder_input = out
        else:
            # Decoder: mapping, assure that input_window_size >= output_window_size
            decoder_output, decoder_hidden = self.rnn_decoder(x, decoder_hidden)
            out = self.l1(decoder_output)  # -> (batch_size, input_window_size, output_vars)
            outputs = out[:, -self.output_window_size:, :]

        return outputs
