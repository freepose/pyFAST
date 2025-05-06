#!/usr/bin/env python
# encoding: utf-8

from typing import Literal

import torch
import torch.nn as nn

from .ar import GAR
from ..base import DirectionalRepresentation, SymmetricAttention
from ..base import MLP
from ...data.scale import InstanceScale, InstanceStandardScale


class COAT(nn.Module):
    """
        Collaborative Attention (COAT).
        Author: Zhijin Wang, Email: zhijinencu@gmail.com

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param mode: set attention mechanism to 'dr', 'sa', or 'dr_sa'.
        :param activation: if True, use 'relu' as activation function.
        :param use_instance_scale: if True, use instance scale (a.k.a., RevIN).
        :param dropout_rate: dropout rate, default is 0.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 mode: Literal['dr', 'sa', 'dr_sa'] = 'dr',
                 activation: str = 'linear', use_instance_scale: bool = False,
                 dropout_rate: float = 0.):
        super(COAT, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.mode = mode

        self.activation = activation
        self.use_instance_scale = use_instance_scale

        self.d1 = nn.Dropout(dropout_rate)

        self.sa_hidden_size = self.input_window_size

        if self.mode == 'dr':
            self.dr0 = DirectionalRepresentation(input_window_size, input_vars, 0, activation, dropout_rate)
            self.dr1 = DirectionalRepresentation(input_window_size, input_vars, 1, activation, dropout_rate)
            self.dr2 = DirectionalRepresentation(input_window_size, input_vars, -1, activation, dropout_rate)
        elif self.mode == 'sa':
            self.sa0 = SymmetricAttention(self.input_vars, self.sa_hidden_size, dim=0)
            self.sa1 = SymmetricAttention(self.input_vars, self.sa_hidden_size, dim=1)
            self.sa2 = SymmetricAttention(self.input_vars, self.sa_hidden_size, dim=2)
        elif self.mode == 'dr_sa':
            self.dr0 = DirectionalRepresentation(input_window_size, input_vars, -1, activation, dropout_rate)
            self.dr1 = DirectionalRepresentation(input_window_size, input_vars, -1, activation, dropout_rate)
            self.dr2 = DirectionalRepresentation(input_window_size, input_vars, -1, activation, dropout_rate)

            self.sa0 = SymmetricAttention(self.input_vars, self.sa_hidden_size, dim=0)
            self.sa1 = SymmetricAttention(self.input_vars, self.sa_hidden_size, dim=1)
            self.sa2 = SymmetricAttention(self.input_vars, self.sa_hidden_size, dim=2)

        self.combination_size = self.input_vars * 4
        self.ar = GAR(self.input_window_size, self.output_window_size)

        self.fc = MLP(self.combination_size, [self.combination_size, self.combination_size // 2], output_vars,
                      False, activation, dropout_rate)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x):
        """ x => [batch_size, input_window_size, input_vars (1)] """
        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x)

        if self.mode == 'dr':
            # x > DR
            out0 = self.dr0(x)  # => [batch_size, input_window_size, input_vars]
            out1 = self.dr1(x)  # => [batch_size, input_window_size, input_vars]
            out2 = self.dr2(x)  # => [batch_size, input_window_size, input_vars]
        elif self.mode == 'sa':
            # x > SA
            out0 = self.sa0(x)  # => [batch_size, input_window_size, input_vars]
            out1 = self.sa1(x)  # => [batch_size, input_window_size, input_vars]
            out2 = self.sa2(x)  # => [batch_size, input_window_size, input_vars]
        else:
            # x > DR > SA
            out0 = self.sa0(self.dr0(x))  # => [batch_size, input_window_size, input_vars]
            out1 = self.sa1(self.dr1(x))  # => [batch_size, input_window_size, input_vars]
            out2 = self.sa2(self.dr2(x))  # => [batch_size, input_window_size, input_vars]

        out = torch.cat([out0, out1, out2, x], dim=2)  # => [batch_size, input_window_size, input_vars * 4]

        # Linearly mapping
        out = self.ar(out)  # => [batch_size, output_window_size, input_vars * 4]
        out = self.fc(out)  # => [batch_size, output_window_size, output_vars]

        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        return out


class CoDR(nn.Module):
    """
        Zhijin Wang, Hanjing Liu, Senzhen Wu, Niansheng Liu, Xiufeng Liu, Yue Hu, and Yonggang Fu. 2024.
        Explainable Time-Varying Directional Representations for Photovoltaic Power Generation Forecasting.
        Journal of Cleaner Production 468 (2024), 143056.
        https://doi.org/10.1016/j.jclepro.2024.143056
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int, output_vars: int,
                 horizon: int = 1, hidden_size: int = 10,
                 use_window_fluctuation_extraction: bool = True, dropout_rate: float = 0.):
        super(CoDR, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.horizon = horizon
        self.hidden_size = hidden_size

        self.use_window_fluctuation_extraction = use_window_fluctuation_extraction

        self.dr0 = DirectionalRepresentation(self.input_window_size, self.input_vars, 0, 'relu', dropout_rate)
        self.dr1 = DirectionalRepresentation(self.input_window_size, self.input_vars, 1, 'relu', dropout_rate)
        self.dr2 = DirectionalRepresentation(self.input_window_size, self.input_vars, 2, 'relu', dropout_rate)

        self.ar0 = GAR(self.input_window_size, self.hidden_size)
        self.ar1 = GAR(self.input_window_size, self.hidden_size)
        self.ar2 = GAR(self.input_window_size, self.hidden_size)
        self.ar3 = GAR(self.input_window_size, self.hidden_size)

        self.ar4 = GAR(self.hidden_size, self.output_window_size)

    def forward(self, x):
        """ x -> [batch_size, input_window_size, input_vars] """

        seq_last = x[:, -self.horizon:, :].detach()  # -> [batch_size, horizon, input_vars]
        seq_last = seq_last.mean(dim=1, keepdim=True)  # -> [batch_size, 1, input_vars]

        # step 1: subtract last
        if self.use_window_fluctuation_extraction:
            x = x - seq_last  # -> [batch_size, input_window_size, input_vars]

        # step 2: representations
        out0 = self.ar0(self.dr0(x))  # -> [batch_size, hidden_size, input_vars]
        out1 = self.ar1(self.dr1(x))  # -> [batch_size, hidden_size, input_vars]
        out2 = self.ar2(self.dr2(x))  # -> [batch_size, hidden_size, input_vars]

        out3 = self.ar3(x)  # -> [batch_size, hidden_size, input_vars]

        # step 3: add back last
        if self.use_window_fluctuation_extraction:
            out3 = out3 + seq_last  # -> [batch_size, hidden_size, input_vars]

        # step 4: linear combination
        out = out0 + out1 + out2 + out3  # -> [batch_size, hidden_size, input_vars]

        # step 5: mapping
        out = self.ar4(out)  # -> [batch_size, output_window_size, input_vars]
        return out


class TCOAT(nn.Module):
    """
        Yue Hu, Hanjing Liu, Senzhen Wu, Yuan Zhao, Zhijin Wang, and Xiufeng Liu. 2024.
        Temporal Collaborative Attention for Wind Power Forecasting.
        Applied Energy 357 (2024), 122502.
        https://doi.org/10.1016/j.apenergy.2023.122502

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: output variables.
        :param rnn_hidden_size: hidden size.
        :param rnn_num_layers: number of layers.
        :param rnn_bidirectional: if True, use bidirectional RNN.
        :param residual_window_size: short-term temporal patterns.
        :param residual_ratio: ratio of residual.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 rnn_hidden_size: int = 64, rnn_num_layers: int = 1, rnn_bidirectional: bool = False,
                 residual_window_size: int = 0, residual_ratio: float = 1., dropout_rate=0.):
        super(TCOAT, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        # RNN
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional

        self.rnn = nn.GRU(input_size=self.input_vars, hidden_size=self.rnn_hidden_size,
                          num_layers=self.rnn_num_layers, bidirectional=self.rnn_bidirectional,
                          batch_first=True, dropout=dropout_rate)

        in_features = self.rnn_hidden_size * 2 if self.rnn_bidirectional else self.rnn_hidden_size
        self.l1 = nn.Linear(in_features, self.input_vars)

        # Residual
        self.residual_window_size = residual_window_size
        self.residual_ratio = residual_ratio

        # DR_SA
        self.dr0 = DirectionalRepresentation(self.input_window_size, self.input_vars, -1, 'relu', dropout_rate)
        self.dr1 = DirectionalRepresentation(self.input_window_size, self.input_vars, -1, 'relu', dropout_rate)
        self.dr2 = DirectionalRepresentation(self.input_window_size, self.input_vars, -1, 'relu', dropout_rate)

        self.sa0 = SymmetricAttention(self.input_vars, self.input_window_size, dim=0)
        self.sa1 = SymmetricAttention(self.input_vars, self.input_window_size, dim=1)
        self.sa2 = SymmetricAttention(self.input_vars, self.input_window_size, dim=2)

        # mappings
        self.ar = GAR(self.input_window_size, self.output_window_size)
        self.l2 = nn.Linear(self.input_vars * 4, self.output_vars)

        # Residual: short-term temporal patterns
        if self.residual_window_size > 0:
            self.residual = GAR(self.residual_window_size, self.output_window_size)

        self.d1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
            x -> (batch_size, input_window_size, input_vars)
         """

        # RNN
        rnn_out, _ = self.rnn(x)  # -> [batch_size, input_window_size, rnn_hidden_size]
        rnn_out = self.l1(rnn_out)  # -> [batch_size, input_window_size, input_vars]

        # DR_SA
        out0 = self.sa0(self.dr0(rnn_out))  # -> [batch_size, input_window_size, input_vars]
        out1 = self.sa1(self.dr1(rnn_out))  # -> [batch_size, input_window_size, input_vars]
        out2 = self.sa2(self.dr2(rnn_out))  # -> [batch_size, input_window_size, input_vars]

        # -> [batch_size, input_window_size, 4 * input_vars]
        out = torch.cat([out0, out1, out2, rnn_out], dim=2)

        out = self.ar(out)  # -> [batch_size, output_window_size, input_vars * 4]
        out = self.l2(out)  # -> [batch_size, output_window_size, output_vars]

        # Residual NN
        if self.residual_window_size > 0:
            z = x[:, -self.residual_window_size:, :]  # -> [batch_size, residual_window_size, input_vars]
            res = self.residual(z) * self.residual_ratio  # -> [batch_size, output_window_size, output_vars]
            out = out + res  # -> [batch_size, output_window_size, output_vars]

        out = self.d1(out)
        # out = out.relu()
        return out


class CTRL(nn.Module):
    """
        Collaborative Temporal Representation Learning (CTRL).
        Author: Zhijin Wang, Email: zhijinecnu@gmail.com

        Y Hu, S Wu, Y Chen, X He, Z Xie, Z Wang, X Liu, Y Fu.
        CTRL: Collaborative Temporal Representation Learning for Wind Power Forecasting.
        ACM EITCE 2024, doi: 10.1145/3711129.371133

        :param input_window_size: input window size.
        :param input_vars:  number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param rnn_hidden_size: hidden size of RNN.
        :param rnn_num_layers: number of layers of RNN.
        :param rnn_bidirectional: if True, use bidirectional RNN.
        :param activation: 'str' type, activation function.
        :param use_instance_scale: if True, use instance scale (RevIN).
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 rnn_hidden_size: int = 32, rnn_num_layers: int = 2, rnn_bidirectional: bool = False,
                 activation: str = 'linear', use_instance_scale: bool = True, dropout_rate: float = 0.):
        super(CTRL, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional
        self.activation = activation
        self.use_instance_scale = use_instance_scale

        # RNN
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn_bidirectional = rnn_bidirectional

        self.rnn = nn.GRU(input_size=self.input_vars, hidden_size=self.rnn_hidden_size,
                          num_layers=self.rnn_num_layers, bidirectional=self.rnn_bidirectional,
                          batch_first=True, dropout=dropout_rate)

        in_features = self.rnn_hidden_size * 2 if self.rnn_bidirectional else self.rnn_hidden_size
        self.l1 = nn.Linear(in_features, self.input_vars)

        # COAT
        self.dr0 = DirectionalRepresentation(self.input_window_size, self.input_vars, 0, activation, dropout_rate)
        self.dr1 = DirectionalRepresentation(self.input_window_size, self.input_vars, 1, activation, dropout_rate)
        self.dr2 = DirectionalRepresentation(self.input_window_size, self.input_vars, 2, activation, dropout_rate)

        self.combination_size = self.input_vars * 4
        self.ar = GAR(self.input_window_size, self.output_window_size)

        self.fc = MLP(self.combination_size, [self.combination_size, self.combination_size // 2], output_vars,
                      False, activation, dropout_rate)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x):
        """ x -> [batch_size, input_window_size, input_vars] """
        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x)

        # rnn as encoder
        rnn_out, _ = self.rnn(x)  # -> [batch_size, input_window_size, hidden_size]
        rnn_out = self.l1(rnn_out)  # -> [batch_size, input_window_size, input_vars]

        # coat as linear mappings: rnn > DR
        out0 = self.dr0(rnn_out)  # => [batch_size, input_window_size, input_vars]
        out1 = self.dr1(rnn_out)  # => [batch_size, input_window_size, input_vars]
        out2 = self.dr2(rnn_out)  # => [batch_size, input_window_size, input_vars]

        out = torch.cat([out0, out1, out2, x], dim=2)  # => [batch_size, input_window_size, comb_size]

        # Linearly mapping to outputs
        out = self.ar(out)  # => [batch_size, output_window_size, comb_size]
        out = self.fc(out)  # => [batch_size, output_window_size, output_vars]

        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        return out
