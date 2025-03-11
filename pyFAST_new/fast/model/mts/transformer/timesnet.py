#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import TokenEmbedding, PositionalEncoding
from ....data import InstanceScale, InstanceStandardScale


class InceptionBlockV1(nn.Module):
    """
        Inception block for TimesNet.

        :param in_channels: input channels.
        :param out_channels: output channels.
        :param num_kernels: number of convolution kernels.
        :param init_weight: whether initialize weights.
    """

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6, init_weight: bool = True):
        super(InceptionBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, input_window_size: int, output_window_size: int,
                 d_model: int, dim_ff: int, num_kernels: int, top_k: int):
        """
            Times block for TimesNet.

            :param input_window_size: input window size.
            :param output_window_size: output window size.
            :param d_model: model dimension, a.k.a., embedding size.
            :param dim_ff: feed forward dimension.
            :param num_kernels: number of convolution kernels.
            :param top_k: the k value.
        """
        super(TimesBlock, self).__init__()
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.d_model = d_model
        self.dim_ff = dim_ff
        self.num_kernels = num_kernels
        self.top_k = top_k

        self.conv = nn.Sequential(InceptionBlockV1(self.d_model, self.dim_ff, self.num_kernels),
                                  nn.GELU(),
                                  InceptionBlockV1(self.dim_ff, self.d_model, self.num_kernels))

    def FFT_for_Period(self, x: torch.Tensor, k: int):
        xf = torch.fft.rfft(x, dim=1)

        # find period by amplitudes
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list

        return period, abs(xf).mean(-1)[:, top_list]

    def forward(self, x: torch.Tensor):
        B, T, N = x.size()
        period_list, period_weight = self.FFT_for_Period(x, self.top_k)

        res = []
        for i in range(self.top_k):
            period = period_list[i]

            # padding
            if (self.input_window_size + self.output_window_size) % period != 0:
                padding_length = (((self.input_window_size + self.output_window_size) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (padding_length - (self.input_window_size + self.output_window_size)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                padding_length = (self.input_window_size + self.output_window_size)
                out = x

            # reshape
            out = out.reshape(B, padding_length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)

            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.input_window_size + self.output_window_size), :])

        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x

        return res


class TimesNet(nn.Module):
    """
        Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long
        TimesNet: Temporal 2D-Variation Modeling for General Series Analysis, ICLR 2023
        url: https://openreview.net/pdf?id=ju_Uqw384Oq

        Official Code: https://github.com/thuml/TimesNet
        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py (our implementation)

        Encoder-only architecture.

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :poram output_vars: output variable number.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_encoder_layers: number of encoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
        :param num_kernels: number of convolution kernels.
        :param top_k: the k value of TimesNet.
    """
    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 d_model: int = 512,
                 num_encoder_layers: int = 1,
                 dim_ff: int = 2048,
                 dropout_rate: float = 0.,
                 num_kernels: int = 6,
                 top_k: int = 5,
                 use_instance_scale: bool = False):
        super(TimesNet, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate
        self.num_kernels = num_kernels
        self.top_k = top_k

        self.encoder_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.encoder_pe = PositionalEncoding(self.d_model)
        self.encoder_dropout = nn.Dropout(dropout_rate)

        self.encoder = nn.ModuleList([TimesBlock(input_window_size, output_window_size, d_model, dim_ff, num_kernels,
                                                 top_k) for _ in range(num_encoder_layers)])

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(input_window_size, input_window_size + output_window_size)
        self.projection = nn.Linear(d_model, output_vars)

        self.inst_scale = InstanceStandardScale() if use_instance_scale else InstanceScale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape is (batch_size, input_window_size, input_vars).
        """

        norm_x = self.inst_scale.fit_transform(x)

        x_embedding = self.encoder_embedding(norm_x) + self.encoder_pe(norm_x)  # -> (batch_size, input_window_size, d_model)

        encoder_in = x_embedding.transpose(1, 2)  # -> (batch_size, d_model, input_window_size)
        encoder_in = self.predict_linear(encoder_in)  # -> (batch_size, d_model, input_window_size + output_window_size)
        encoder_out = encoder_in.transpose(1, 2)  # -> (batch_size, input_window_size + output_window_size, d_model)

        for i in range(self.num_encoder_layers):
            encoder_out = self.encoder[i](encoder_out)
            encoder_out = self.layer_norm(encoder_out)

        decoder_out = self.projection(encoder_out)  # -> (batch_size, input_window_size + output_window_size, output_vars)

        out = self.inst_scale.inverse_transform(decoder_out)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, input_vars)

        return out

