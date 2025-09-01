#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from ...data import InstanceStandardScale   # RevIN: Reversible Instance Normalization
from ..base.decomposition import DecomposeSeries


class Amplifier(nn.Module):
    """
        Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting.
        Jingru Fei, Kun Yi, Wei Fan, Qi Zhang, Zhendong Niu.
        https://arxiv.org/abs/2501.17216

        Authors' implementation: https://github.com/aikunyi/Amplifier

        This model has been changed to support both odd and even input/output window sizes.

        The ``output_window_size`` === ``input_window_size``.

        :param input_window_size: input sequence length.
        :param input_vars: number of input variables (channels).
        :param output_window_size: output sequence length.
        :param kernel_size: size of the kernel for the decomposition block.
        :param hidden_size: size of the hidden layer in the linear layers.
        :param use_sci_block: whether to use the SCI block for extracting common and specific patterns.
        :param use_instance_scale: whether to use instance standard scaling for input normalization.
    """
    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1,
                 kernel_size: int = 25, hidden_size: int = 128, use_sci_block: bool = True,
                 use_instance_scale: bool = True):
        super(Amplifier, self).__init__()

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.input_vars = input_vars

        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.use_sci_block = use_sci_block
        self.use_instance_scale = use_instance_scale

        self.decompsition = DecomposeSeries(kernel_size)

        self.input_freq_len = input_window_size // 2 + 1
        self.output_freq_len = output_window_size // 2 + 1

        self.mask_matrix = nn.Parameter(torch.ones(self.input_freq_len, self.input_vars))

        self.freq_linear_real = nn.Linear(self.input_freq_len, self.output_freq_len)
        self.freq_linear_imag = nn.Linear(self.input_freq_len, self.output_freq_len)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.input_window_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_window_size)
        )

        self.linear_trend = nn.Sequential(
            nn.Linear(self.input_window_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_window_size)
        )

        if self.use_sci_block:
            self.extract_common_pattern = nn.Sequential(
                nn.Linear(self.input_vars, self.input_vars),
                nn.LeakyReLU(),
                nn.Linear(self.input_vars, 1)
            )

            self.model_common_pattern = nn.Sequential(
                nn.Linear(self.input_window_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.input_window_size)
            )

            self.model_spacific_pattern = nn.Sequential(
                nn.Linear(self.input_window_size, self.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_size, self.input_window_size)
            )

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
        """
        B, T, C = x.size()

        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x)

        # Energy Amplification Block
        x_fft = torch.fft.rfft(x, dim=1)
        x_inverse_fft = torch.flip(x_fft, dims=[1]) * self.mask_matrix
        x_amplifier_fft = x_fft + x_inverse_fft
        x_amplifier = torch.fft.irfft(x_amplifier_fft, n=self.input_window_size, dim=1)

        if self.use_sci_block:
            x = x_amplifier
            common_pattern = self.extract_common_pattern(x)
            common_pattern = self.model_common_pattern(common_pattern.permute(0, 2, 1)).permute(0, 2, 1)
            specific_pattern = x - common_pattern.repeat(1, 1, C)
            specific_pattern = self.model_spacific_pattern(specific_pattern.permute(0, 2, 1)).permute(0, 2, 1)
            x = specific_pattern + common_pattern.repeat(1, 1, C)
            x_amplifier = x

        # Seasonal Trend Forecaster
        trend, residuals = self.decompsition(x_amplifier)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        residuals = self.linear_seasonal(residuals.permute(0, 2, 1)).permute(0, 2, 1)
        out_amplifier = trend + residuals

        # Energy Restoration Block
        out_amplifier_fft = torch.fft.rfft(out_amplifier, dim=1)

        # x_inverse_fft = self.freq_linear(x_inverse_fft.permute(0, 2, 1)).permute(0, 2, 1)

        real_out = self.freq_linear_real(x_inverse_fft.real.permute(0, 2, 1)).permute(0, 2, 1)
        imag_out = self.freq_linear_imag(x_inverse_fft.imag.permute(0, 2, 1)).permute(0, 2, 1)
        x_inverse_fft = torch.complex(real_out, imag_out)

        out_fft = out_amplifier_fft - x_inverse_fft
        out = torch.fft.irfft(out_fft, n=self.output_window_size, dim=1)

        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        return out
