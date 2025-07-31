#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from typing import Literal
from ...data import InstanceStandardScale
from ..base.decomposition import DecomposeSeries
from .ar import GAR, AR, VAR, ANN


class RLinear(nn.Module):
    """

        Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping.
        Zhe Li, Shiyi Qi, Yiduo Li, Zenglin Xu.
        Paper url: https://arxiv.org/pdf/2305.10721

        Author provided code: https://github.com/plumprc/RTSF

        :param input_window_size: input sequence length.
        :param input_vars: number of input variables (channels).
        :param output_window_size: output sequence length.
        :param dropout_rate: dropout rate applied to the input.
        :param use_instance_scale: whether to use instance standard scaling.
        :param mapping: the mapping method, can be 'gar' (Global AR), 'ar' (AR),
                        or 'mlp' (Multi-layer Perceptron).
                        GAR means all variables share the same AR model,
                        AR means each variable has its own AR model,
                        and MLP means a multi-layer perceptron is used to map the input window to the output window.
        :param d_model: the dimension of the model, required when mapping is 'mlp'.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 dropout_rate: float = 0.0, use_instance_scale: bool = True,
                 mapping: Literal['gar', 'ar', 'mlp'] = 'gar', d_model: int = None):
        super(RLinear, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size

        self.dropout_rate = dropout_rate
        self.use_instance_scale = use_instance_scale
        self.mapping = mapping

        if self.mapping == 'gar':
            self.l1 = GAR(self.input_window_size, self.output_window_size)
        elif self.mapping == 'ar':
            self.l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
        elif self.mapping == 'mlp':
            assert d_model is not None, "d_model must be provided when ``mapping`` is 'mlp'."
            self.l1 = nn.Sequential(
                ANN(self.input_window_size, self.input_window_size, d_model),
                GAR(self.input_window_size, self.output_window_size)
            )

        self.d1 = nn.Dropout(self.dropout_rate)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
            :param x_mask: shape is (batch_size, input_window_size, input_vars), mask tensor.
        """

        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x, x_mask)

        if x_mask is not None:
            x[~x_mask] = 0.

        x = self.d1(x)
        out = self.l1(x)  # -> (batch_size, output_window_size, input_vars)

        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        return out


class STD(nn.Module):
    """
        Seasonal-Trend Decomposition.
        Author provided code: https://github.com/plumprc/RTSF

        :param input_window_size: input window size, a.k.a., input sequence length.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param d_model: the dimension of the model.
        :param kernel_size: the kernel size of series decomposition function.
        :param use_instance_scale: whether to use instance standard scaling.
                                   if False, then the model is equivalent to ** DLinear **.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 kernel_size: int = 25, d_model: int = 128, use_instance_scale: bool = False):
        super(STD, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size

        self.kernel_size = kernel_size
        self.d_model = d_model
        self.use_instance_scale = use_instance_scale

        self.decomposition = DecomposeSeries(self.kernel_size)
        self.l1 = GAR(self.input_window_size, self.output_window_size)
        self.l2 = ANN(self.input_window_size, self.output_window_size, self.d_model)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x):
        """
            :param x: Input tensor shape is (batch_size, input_window_size, input_vars).
        """
        trend_init, seasonal_init = self.decomposition(x)   # -> (batch_size, input_window_size, input_vars)

        if self.use_instance_scale:
            trend_init = self.inst_scaler.fit_transform(trend_init)

        seasonal_init = self.l1(seasonal_init)  # -> (batch_size, output_window_size, input_vars)
        trend_init = self.l2(trend_init)        # -> (batch_size, output_window_size, input_vars)

        if self.use_instance_scale:
            trend_init = self.inst_scaler.inverse_transform(trend_init)

        out = seasonal_init + trend_init

        return out
