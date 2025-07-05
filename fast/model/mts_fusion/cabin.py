#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from typing import Literal

from ..mts import GAR, KAN


class AmbientRepresentationModule(nn.Module):
    def __init__(self, window_size: int, input_size: int, dim: int = 0,
                 use_activate: bool = True, dropout: float = 0.1):
        super(AmbientRepresentationModule, self).__init__()

        self.window_size = window_size
        self.input_size = input_size
        self.dim = dim
        self.use_activate = use_activate

        self.weight = nn.Parameter(torch.zeros(self.window_size, self.input_size))
        nn.init.uniform_(self.weight, a=0.01, b=0.1)

    def forward(self, x: torch.Tensor):
        if self.use_activate:
            x = torch.relu(x)

        out = x * self.weight

        if self.dim != -1:
            out = out.softmax(dim=self.dim)

        return out


class CollaborationAmbientVariables(nn.Module):
    def __init__(self, input_window_size: int, output_window_size: int,
                 input_size: int, hidden_size: int, output_size: int, use_activate: bool = True):
        super(CollaborationAmbientVariables, self).__init__()

        self.use_activate = use_activate
        self.mapping = nn.Linear(input_size, hidden_size)
        self.ar = GAR(input_window_size, output_window_size)
        self.kan = KAN(output_window_size, hidden_size, output_window_size, output_size,
                      layers_hidden=[hidden_size // 2], patch_form=False)

    def forward(self, x: torch.Tensor):
        out = self.mapping(x)

        if self.use_activate:
            out = torch.relu(out)

        out = self.ar(out)

        if self.use_activate:
            out = torch.relu(out)

        out = self.kan(out)

        return out


class Cabin(nn.Module):
    """
        Senzhen Wu, Yu Chen, Xinhao He, Zhijin Wang, Xiufeng Liu, Yonggang Fu

        :param input_window_size: input sequence length.
        :param input_vars: number of input variables (channels).
        :param output_window_size: output sequence length.
        :param output_vars: number of output variables (channels).
        :param ex_retain_window_size: input window size of exogenous variable.
        :param ex_vars: number of exogenous variables.
        :param hidden_size: the hidden size of CAV layers.
        :param use_activation: whether use ReLU as activation.
        :param dropout_rate: dropout rate.
        :param mode: the fusion mode.
        :param use_temporal_softmax: whether use temporal softmax.
        :param use_feature_wise_softmax: whether use feature wise softmax.
        :param use_sample_softmax: whether use sample wise softmax.
    """

    def __init__(self, input_window_size: int = 0, input_vars: int = 0,
                 output_window_size: int = 1, output_vars: int = 1,
                 ex_retain_window_size: int = 0, ex_vars: int = 0,
                 hidden_size: int = 64, use_activation: bool = True, dropout_rate: float = 0.,
                 mode: Literal['data_first', 'learning_first', 'only_target'] = 'data_first',
                 use_temporal_softmax: bool = True,
                 use_feature_wise_softmax: bool = True,
                 use_sample_softmax: bool = True):
        super(Cabin, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        self.hidden_size = hidden_size
        self.mode = mode
        self.use_arm0 = use_sample_softmax
        self.use_arm1 = use_temporal_softmax
        self.use_arm2 = use_feature_wise_softmax

        # Debug info
        # print(f"""
        # Mode: {mode}
        # Use Temporal Softmax: {use_temporal_softmax}
        # Use Feature-wise Softmax: {use_feature_wise_softmax}
        # Use Sample Softmax: {use_sample_softmax}
        # Hidden Size: {hidden_size}
        # """)

        if self.mode == 'data_first':
            self.arm0 = AmbientRepresentationModule(self.input_window_size, self.input_vars + self.ex_vars, 0, use_activation, dropout_rate)
            self.arm1 = AmbientRepresentationModule(self.input_window_size, self.input_vars + self.ex_vars, 1, use_activation, dropout_rate)
            self.arm2 = AmbientRepresentationModule(self.input_window_size, self.input_vars + self.ex_vars, 2, use_activation, dropout_rate)

            self.cav = CollaborationAmbientVariables(self.input_window_size, self.output_window_size, (self.input_vars + self.ex_vars) * (1 + self.use_arm0 + self.use_arm1 + self.use_arm2), self.hidden_size, self.output_vars, use_activation)

        elif self.mode == 'learning_first':
            self.arm0 = AmbientRepresentationModule(self.input_window_size, self.input_vars, 0, use_activation, dropout_rate)
            self.arm1 = AmbientRepresentationModule(self.input_window_size, self.input_vars, 1, use_activation, dropout_rate)
            self.arm2 = AmbientRepresentationModule(self.input_window_size, self.input_vars, 2, use_activation, dropout_rate)

            self.ex_arm0 = AmbientRepresentationModule(self.input_window_size, self.ex_vars, 0, use_activation, dropout_rate)
            self.ex_arm1 = AmbientRepresentationModule(self.input_window_size, self.ex_vars, 1, use_activation, dropout_rate)
            self.ex_arm2 = AmbientRepresentationModule(self.input_window_size, self.ex_vars, 2, use_activation, dropout_rate)

            self.cav = CollaborationAmbientVariables(self.input_window_size, self.output_window_size, (self.input_vars + self.ex_vars) * (1 + self.use_arm0 + self.use_arm1 + self.use_arm2), self.hidden_size, self.output_vars, use_activation)

        elif self.mode == 'only_target':
            self.arm0 = AmbientRepresentationModule(self.input_window_size, self.input_vars, 0, use_activation, dropout_rate)
            self.arm1 = AmbientRepresentationModule(self.input_window_size, self.input_vars, 1, use_activation, dropout_rate)
            self.arm2 = AmbientRepresentationModule(self.input_window_size, self.input_vars, 2, use_activation, dropout_rate)

            self.cav = CollaborationAmbientVariables(self.input_window_size, self.output_window_size, self.input_vars * (1 + self.use_arm0 + self.use_arm1 + self.use_arm2), self.hidden_size, self.output_vars, use_activation)

        else:
            raise ValueError('Unknow mode.')

    def forward(self, x: torch.Tensor, e: torch.Tensor = None):
        """
            :param e: shape is [batch_size, ex_input_window_size, ex_input_size]
            :param x: shape is [batch_size, input_window_size, input_size]
            :return:  shape is [batch_size, output_window_size, output_size]
        """

        if self.mode == 'data_first':
            out = torch.cat([x, e], dim=2)

            cat_list = []
            if self.use_arm0:
                out0 = self.arm0(out)
                cat_list.append(out0)
            if self.use_arm1:
                out1 = self.arm1(out)
                cat_list.append(out1)
            if self.use_arm2:
                out2 = self.arm2(out)
                cat_list.append(out2)
            cat_list.append(out)

            out = torch.cat(cat_list, dim=2)

            out = self.cav(out)

            return out

        elif self.mode == 'learning_first':
            cat_list, ex_cat_list = [], []
            if self.use_arm0:
                out0 = self.arm0(x)
                ex_out0 = self.ex_arm0(e)
                cat_list.append(out0)
                ex_cat_list.append(ex_out0)
            if self.use_arm1:
                out1 = self.arm1(x)
                ex_out1 = self.ex_arm1(e)
                cat_list.append(out1)
                ex_cat_list.append(ex_out1)
            if self.use_arm2:
                out2 = self.arm2(x)
                ex_out2 = self.ex_arm2(e)
                cat_list.append(out2)
                ex_cat_list.append(ex_out2)
            cat_list.append(x)
            ex_cat_list.append(e)

            out = torch.cat(cat_list + ex_cat_list, dim=2)

            out = self.cav(out)

            return out

        elif self.mode == 'only_target':
            cat_list = []
            if self.use_arm0:
                out0 = self.arm0(x)
                cat_list.append(out0)
            if self.use_arm1:
                out1 = self.arm1(x)
                cat_list.append(out1)
            if self.use_arm2:
                out2 = self.arm2(x)
                cat_list.append(out2)
            cat_list.append(x)

            out = torch.cat(cat_list, dim=2)

            out = self.cav(out)

            return out


        return None
