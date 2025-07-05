#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class MinLSTMCell(nn.Module):
    """
        Were RNNs All We Needed?
        Leo Feng, Frederick Tung, Mohamed Osama Ahmed, Yoshua Bengio, Hossein Hajimirsadegh.
        arXiv 2024.
        url: https://arxiv.org/pdf/2410.01201
        code: https://github.com/axion66/minLSTM-implementation

        :param input_size: The number of features in the input.
        :param hidden_size: The number of features in the hidden state.
        :param bias: If False, disables the use of bias in the linear layers.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False):
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
