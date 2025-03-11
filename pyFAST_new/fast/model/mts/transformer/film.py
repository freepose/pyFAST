#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from scipy import special as ss

from ....data import InstanceScale, InstanceStandardScale


def transition(N):
    Q = np.arange(N, dtype=np.float64)
    R = (2 * Q + 1)[:, None]  # / theta
    j, i = np.meshgrid(Q, Q)
    A = np.where(i < j, -1, (-1.) ** (i - j + 1)) * R
    B = (-1.) ** Q[:, None] * R
    return A, B


class HiPPO_LegT(nn.Module):
    """
        :param N: the order of the HiPPO projection.
        :param dt: discretization step size - should be roughly inverse to the length of the sequence.
    """

    def __init__(self, N: int, dt: float, discretization: str = 'bilinear'):
        super(HiPPO_LegT, self).__init__()

        self.N = N
        A, B = transition(N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A))
        self.register_buffer('B', torch.Tensor(B))
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix', torch.Tensor(ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T))

    def forward(self, inputs: torch.Tensor):
        c = torch.zeros(inputs.shape[:-1] + tuple([self.N]), device=inputs.device)
        cs = []
        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            new = f @ self.B.unsqueeze(0)
            c = F.linear(c, self.A) + new
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c: torch.Tensor):
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, seq_len: int):
        """
            1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = min(32, seq_len // 2)
        self.index = list(range(0, self.modes))

        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        self.weights_imag = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))

    def compl_mul1d(self, order: str, x: torch.Tensor, weights_real: nn.Parameter, weights_imag: nn.Parameter):
        return torch.complex(torch.einsum(order, x.real, weights_real) - torch.einsum(order, x.imag, weights_imag),
                             torch.einsum(order, x.real, weights_imag) + torch.einsum(order, x.imag, weights_real))

    def forward(self, x: torch.Tensor):
        B, H, E, N = x.shape
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(B, H, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        a = x_ft[:, :, :, :self.modes]
        out_ft[:, :, :, :self.modes] = self.compl_mul1d("bjix,iox->bjox", a, self.weights_real, self.weights_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FiLM(nn.Module):
    """
        Tian Zhou, Ziqing Ma, Xue wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, Rong Jin
        FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting, NeurIPS 2022
        url: https://arxiv.org/abs/2205.08897

        Official Code: https://github.com/tianzhou2011/FiLM/
        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/FiLM.py (our implementation)

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param use_instance_scale: whether to use instance standard scale (a.k.a., RevIN).
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 d_model: int = 512, use_instance_scale: bool = True):
        super(FiLM, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.d_model = d_model

        self.affine_weight = nn.Parameter(torch.ones(1, 1, input_vars))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, input_vars))

        self.multiscale = [1, 2, 4]
        self.window_size = [256]
        self.legts = nn.ModuleList([HiPPO_LegT(n, 1. / output_window_size / i)
                                    for n in self.window_size
                                    for i in self.multiscale])
        self.spec_conv_1 = nn.ModuleList([SpectralConv1d(n, n, min(output_window_size, input_window_size))
                                          for n in self.window_size
                                          for _ in range(len(self.multiscale))])
        self.mlp = nn.Linear(len(self.multiscale) * len(self.window_size), 1)

        self.inst_scale = InstanceStandardScale(input_vars) if use_instance_scale else InstanceScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x -> (batch_size, input_window_size, input_vars) """

        norm_x = self.inst_scale.fit_transform(x)
        x_encoder = norm_x

        x_decs = []
        jump_dist = 0
        for i in range(0, len(self.multiscale) * len(self.window_size)):
            x_in_len = self.multiscale[i % len(self.multiscale)] * self.output_window_size
            x_in = x_encoder[:, -x_in_len:]
            legt = self.legts[i]
            x_in_c = legt(x_in.transpose(1, 2)).permute([1, 2, 3, 0])[:, :, :, jump_dist:]
            out1 = self.spec_conv_1[i](x_in_c)
            if self.input_window_size >= self.output_window_size:
                x_dec_c = out1.transpose(2, 3)[:, :, self.output_window_size - 1 - jump_dist, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:, :, -1, :]
            x_dec = x_dec_c @ legt.eval_matrix[-self.output_window_size:, :].T
            x_decs.append(x_dec)

        x_decoder = torch.stack(x_decs, dim=-1)
        x_decoder = self.mlp(x_decoder).squeeze(-1).permute(0, 2, 1)

        out = self.inst_scale.inverse_transform(x_decoder)

        return out
