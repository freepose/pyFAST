#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Optional
from ..metric import MAE
from ..data.utils import compute_deltas


class TemporalDecay(nn.Module):
    """
    时间衰减模块

    计算公式: gamma = exp(-max(0, W * delta + b))
    其中 delta 是时间间隔

    Args:
        input_size: 输入维度
        output_size: 输出维度
        diag: 是否使用对角矩阵（特征独立衰减）
    """

    def __init__(self, input_size: int, output_size: int, diag: bool = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag
        self.input_size = input_size
        self.output_size = output_size

        if self.diag:
            # 对角矩阵：每个特征独立衰减
            assert input_size == output_size

        self.decay = nn.Linear(input_size, output_size, bias=True)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta: [batch_size, input_size] 或 [batch_size, seq_len, input_size] 时间间隔

        Returns:
            gamma: 对应形状的衰减因子
        """

        # 计算 W * delta + b，然后应用 exp(-max(0, x))
        gamma = torch.exp(-torch.relu(self.decay(delta)))

        return gamma


class RITS(nn.Module):
    def __init__(self, input_vars: int, rnn_hidden_size: int = 32,
                 n_classes: int = None, dropout_rate: float = 0.25,
                 recovery_weight: float = 0.1, fill_na: str = None):
        super(RITS, self).__init__()

        # Ensure input_vars == output_vars, the number of features of time series, they are values, mask, deltas
        self.input_vars = input_vars
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.recovery_weight = recovery_weight
        self.fill_na = fill_na

        self.rnn_cell = nn.LSTMCell(input_vars * 2, self.rnn_hidden_size)  # Why input_size * 2?

        self.temp_decay_h = TemporalDecay(input_size=input_vars, output_size=self.rnn_hidden_size, diag=False)
        self.temp_decay_x = TemporalDecay(input_size=input_vars, output_size=input_vars, diag=True)

        self.hist_reg = nn.Linear(self.rnn_hidden_size, input_vars)
        self.feat_reg = nn.Linear(input_vars, input_vars)

        self.weight_combine = nn.Linear(input_vars * 2, input_vars)  # why input_size * 2?

        self.dropout = nn.Dropout(dropout_rate)

        if self.n_classes is not None and self.n_classes > 0:
            self.out = nn.Linear(self.rnn_hidden_size, self.n_classes)

        self.input_aware_loss = 0.  # recovery loss
        self.input_aware_loss_fn = MAE()

    def forward(self, values: torch.Tensor, masks: torch.Tensor, deltas: Optional[torch.Tensor] = None):
        """
            :param values: shape is ``(batch_size, seq_len, input_size)``
            :param masks: shape is ``(batch_size, seq_len, input_size)``
            :param deltas: shape is ``(batch_size, seq_len, input_size)``
        """

        batch_size, seq_len, input_size = values.size()

        h = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        c = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)

        x_h_list, z_h_list, c_h_list = [], [], []
        imputations = []
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

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)  # mask MAE on (x, x_h) at time t
            x_h_list.append(x_h.unsqueeze(1))

            x_c = m * x + (1 - m) * x_h
            z_h = self.feat_reg(x_c)  # mask MAE on (x, z_h) at time t
            z_h_list.append(z_h.unsqueeze(1))

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=1))
            c_h = alpha * z_h + (1 - alpha) * x_h  # mask MAE on (x, c_h) at time t
            c_h_list.append(c_h.unsqueeze(1))

            c_c = m * x + (1 - m) * c_h
            inputs = torch.cat([c_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(c_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)
        x_hs = torch.cat(x_h_list, dim=1)
        z_hs = torch.cat(z_h_list, dim=1)
        c_hs = torch.cat(c_h_list, dim=1)

        # compute input-aware loss
        self.input_aware_loss = self.input_aware_loss_fn(values, x_hs, masks) + \
                                self.input_aware_loss_fn(values, z_hs, masks) + \
                                self.input_aware_loss_fn(values, c_hs, masks)
        self.input_aware_loss *= self.recovery_weight

        if self.n_classes is not None and self.n_classes > 0:
            y_h = self.out(h)
            return y_h, imputations

        return imputations


class RITSI(nn.Module):

    def __init__(self, input_size: int, rnn_hidden_size: int = 32,
                 n_classes: int = None,
                 recovery_weight: float = 0.1, fill_na: str = None):
        super(RITSI, self).__init__()

        self.input_size = input_size  # Ensure input_size == output_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.recovery_weight = recovery_weight
        self.fill_na = fill_na

        self.rnn_cell = nn.LSTMCell(self.input_size * 2, self.rnn_hidden_size)

        self.temp_decay = TemporalDecay(input_size=self.input_size, output_size=self.rnn_hidden_size, diag=False)
        self.regression = nn.Linear(self.rnn_hidden_size, self.input_size)

        if self.n_classes is not None and self.n_classes > 0:
            self.out = nn.Linear(self.rnn_hidden_size, n_classes)

        self.input_aware_loss = 0.
        self.input_aware_loss_fn = MAE()

    def forward(self, values: torch.Tensor, masks: torch.Tensor, deltas: Optional[torch.Tensor] = None):
        """
            :param values: shape is ``(batch_size, seq_len, input_size)``
            :param masks: shape is ``(batch_size, seq_len, input_size)``
            :param deltas: shape is ``(batch_size, seq_len, input_size)``, optional

        """
        batch_size, seq_len, input_size = values.size()

        h = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)
        c = torch.zeros(batch_size, self.rnn_hidden_size, device=values.device)

        x_c_list = []
        imputations = []
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

            gamma = self.temp_decay(d)
            h = h * gamma
            x_h = self.regression(h)

            x_c = m * x + (1 - m) * x_h  # mask MAE on (x, x_c) at time t
            x_c_list.append(x_c.unsqueeze(dim=1))

            inputs = torch.cat([x_c, m], dim=1)
            h, c = self.rnn_cell(inputs, (h, c))
            imputations.append(x_c.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        x_cs = torch.cat(x_c_list, dim=1)
        self.input_aware_loss = self.input_aware_loss_fn(values, x_cs, masks) * self.recovery_weight

        if self.n_classes is not None and self.n_classes > 0:
            y_h = self.out(h)
            return y_h, imputations

        return imputations
