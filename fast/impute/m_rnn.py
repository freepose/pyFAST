#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rits import TemporalDecay
from ..metric import MAE


class MRNN(nn.Module):
    def __init__(self, input_size: int, rnn_hidden_size: int = 32, n_classes: int = 2,
                 dropout_rate: float = 0.25, recovery_weight: float = 0.1):
        super(MRNN, self).__init__()

        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.recovery_weight = recovery_weight

        self.rnn_cell = nn.LSTMCell(self.input_size * 3, self.rnn_hidden_size)
        self.pred_rnn = nn.LSTM(self.input_size, self.rnn_hidden_size, batch_first=True)

        self.temp_decay_h = TemporalDecay(input_size=self.input_size, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=self.input_size, output_size=self.input_size, diag=True)

        self.hist_reg = nn.Linear(self.rnn_hidden_size * 2, 35)
        self.feat_reg = nn.Linear(self.input_size, self.input_size)

        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(self.rnn_hidden_size, self.n_classes)

        self.input_aware_loss = 0.
        self.input_aware_loss_fn = MAE()

    def get_hidden(self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor):
        batch_size, seq_len, input_size = values.size()

        hidden_states = []

        h = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        c = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)

        for t in range(seq_len):
            hidden_states.append(h)

            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            inputs = torch.cat([x, m, d], dim=1)

            h, c = self.rnn_cell(inputs, (h, c))

        return hidden_states

    def forward(self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor,
                backward_values: torch.Tensor,
                backward_masks: torch.Tensor,
                backward_deltas: torch.Tensor):

        batch_size, seq_len, input_size = values.size()

        hidden_forward = self.get_hidden(values, masks, deltas)
        hidden_backward = self.get_hidden(backward_values, backward_masks, backward_deltas)[::-1]

        impute_list = []
        x_c_list = []
        for t in range(seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            hf = hidden_forward[t]
            hb = hidden_backward[t]
            h = torch.cat([hf, hb], dim=1)

            x_h = self.hist_reg(h)
            x_f = self.feat_reg(x)

            alpha = F.sigmoid(self.weight_combine(torch.cat([m, d], dim=1)))
            x_c = alpha * x_h + (1 - alpha)  # mask MAE on (x, c_h) at time t
            x_c_list.append(x_c.unsqueeze(1))

            impute_list.append(x_c.unsqueeze(dim=1))

        impute_tensor = torch.cat(impute_list, dim=1)
        x_cs = torch.cat(x_c_list, dim=1)
        self.input_aware_loss = self.input_aware_loss_fn(values, x_cs, masks) * self.recovery_weight

        # h = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        # c = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        out, (h, c) = self.pred_rnn(impute_tensor.detach().data)
        y_h = self.out(h.squeeze(0))

        return y_h, impute_tensor
