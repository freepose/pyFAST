#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Optional
from .rits import TemporalDecay
from ..data.utils import compute_deltas


class GRUD(nn.Module):
    """
        GRU-D: Gated Recurrent Unit with Decay for Multivariate Time Series Imputation.

        Che, Zhengping, et al.
        "Recurrent neural networks for multivariate time series with missing values."
        Scientific reports 8.1 (2018): 1-12.

        :param input_vars: the number of input variables.
        :param rnn_hidden_size: the hidden size of RNN.
        :param n_classes: the number of classes for classification tasks. If None, the model
                            only performs imputation.
        :param dropout_rate: the dropout rate.
    """
    def __init__(self, input_vars: int, rnn_hidden_size: int = 32,
                 n_classes: Optional[int] = None,
                 dropout_rate: float = 0.5, fill_na: str = None):
        super(GRUD, self).__init__()

        self.input_vars = input_vars
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.fill_na = fill_na

        self.rnn_cell = nn.LSTMCell(self.input_vars * 2, self.rnn_hidden_size)

        self.temp_decay_h = TemporalDecay(input_size=self.input_vars, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.input_vars, output_size=self.input_vars, diag=True)

        self.hist_reg = nn.Linear(self.rnn_hidden_size, self.input_vars)
        self.feat_reg = nn.Linear(self.input_vars, self.input_vars)

        self.weight_combine = nn.Linear(self.input_vars * 2, self.input_vars)

        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(self.rnn_hidden_size, self.n_classes) if self.n_classes is not None else None

        self.output_window_size = None  # for update during training

    def forward(self, values: torch.Tensor, masks: torch.Tensor,
                deltas: Optional[torch.Tensor] = None, forwards: Optional[torch.Tensor] = None):
        """
            :param values: shape is ``(batch_size, seq_len, input_size)``
            :param masks: shape is ``(batch_size, seq_len, input_size)``
            :param deltas: shape is ``(batch_size, seq_len, input_size)``
            :param forwards: shape is ``(batch_size, seq_len, input_size)``, optional
        """
        batch_size, seq_len, input_size = values.size()

        h = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        c = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)

        impute_list = []
        masks_float = masks.float()

        if self.fill_na is not None:
            if self.fill_na == 'zero':
                values[~masks] = 0.

        if deltas is None:
            deltas = compute_deltas(masks_float)  # shape is (batch_size, seq_len, input_size)

        for t in range(seq_len):
            x = values[:, t, :]
            m = masks_float[:, t, :]
            d = deltas[:, t, :]
            f = forwards[:, t, :] if forwards is not None else 1.

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = m * x + (1 - m) * (1 - gamma_x) * f
            inputs = torch.cat([x_h, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            impute_list.append(x_h.unsqueeze(dim=1))

        # If output_window_size not set by Trainer, default to 1
        # if self.output_window_size is None:
        #     self.output_window_size = seq_len

        impute_tensor = torch.cat(impute_list, dim=1)
        impute_tensor = impute_tensor[:, -self.output_window_size:, :]

        if self.n_classes is not None:
            y_h = self.out(self.dropout(h))
            return y_h, impute_tensor
        else:
            return impute_tensor
