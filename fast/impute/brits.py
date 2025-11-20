#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from .rits import RITSI, RITS


class BRITSI(nn.Module):

    def __init__(self, input_size: int, rnn_hidden_size: int = 32, n_classes: int = 2,
                 recovery_weight: float = 0.1, consistency_weight: float = 0.1):
        super(BRITSI, self).__init__()

        # Ensure input_size == output_size, the number of features of time series, they are values, mask, deltas
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.recovery_weight = recovery_weight
        self.consistency_weight = consistency_weight

        self.ritsi_f = RITSI(self.input_size, self.rnn_hidden_size, self.n_classes, self.recovery_weight)
        self.ritsi_b = RITSI(self.input_size, self.rnn_hidden_size, self.n_classes, self.recovery_weight)

        self.input_aware_loss = 0.

    def forward(self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor,
                backward_values: torch.Tensor,
                backward_masks: torch.Tensor,
                backward_deltas: torch.Tensor):

        y_hat_f, impute_f = self.ritsi_f(values, masks, deltas)
        y_hat_b, impute_b = self.ritsi_b(backward_values, backward_masks, backward_deltas)
        impute_b = torch.flip(impute_b, dims=[1])

        y_hat = (y_hat_f + y_hat_b) / 2
        impute = (impute_f + impute_b) / 2

        self.input_aware_loss = self.ritsi_f.input_aware_loss + self.ritsi_b.input_aware_loss
        self.input_aware_loss += torch.abs(impute_f - impute_b).mean() * self.consistency_weight

        return y_hat_f, y_hat_b, y_hat, impute


class BRITS(nn.Module):
    def __init__(self, input_size: int, rnn_hidden_size: int = 32, n_classes: int = 2,
                 dropout_rate: float = 0.25,
                 recovery_weight: float = 0.1):
        super(BRITS, self).__init__()

        # Ensure input_size == output_size, the number of features of time series, they are values, mask, deltas
        self.input_size = input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.n_classes = n_classes
        self.recovery_weight = recovery_weight

        self.rits_f = RITS(self.input_size, self.rnn_hidden_size, self.n_classes, dropout_rate, self.recovery_weight)
        self.rits_b = RITS(self.input_size, self.rnn_hidden_size, self.n_classes, dropout_rate, self.recovery_weight)

        self.input_aware_loss = 0.

    def forward(self, values: torch.Tensor, masks: torch.Tensor, deltas: torch.Tensor,
                backward_values: torch.Tensor,
                backward_masks: torch.Tensor,
                backward_deltas: torch.Tensor):
        y_hat_f, impute_f = self.rits_f(values, masks, deltas)
        y_hat_b, impute_b = self.rits_b(backward_values, backward_masks, backward_deltas)
        impute_b = torch.flip(impute_b, dims=[1])

        y_hat = (y_hat_f + y_hat_b) / 2
        impute = (impute_f + impute_b) / 2

        self.input_aware_loss = self.rits_f.input_aware_loss + self.rits_b.input_aware_loss

        return y_hat_f, y_hat_b, y_hat, impute
