#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class SNorm(nn.Module):
    def __init__(self, channels: int):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self, num_nodes: int, channels: int, track_running_stats: bool = True, momentum: float = 0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x: torch.Tensor):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class STNorm(nn.Module):
    """
        Jinliang Deng, Xiusi Chen, Renhe Jiang, Xuan Song, Ivor W. Tsang
        ST-Norm: Spatial and Temporal Normalization for Multi-variate Time Series Forecasting
        KDD 2021

        url: https://dl.acm.org/doi/10.1145/3447548.3467330
        Official Code: https://github.com/JLDeng/ST-Norm/blob/master/models/Wavenet.py

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_window_size: Number of time steps in output window
        :param tnorm_bool: enable temporal normalization.
        :param snorm_bool: enable spatial normalization.
        :param channels: number of convolutional channels.
        :param kernel_size: dilation kernel size.
        :param blocks: number of residual blocks.
        :param layers: number of layers per block.
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int,
                 tnorm_bool: bool = True, snorm_bool: bool = True,
                 channels: int = 16, kernel_size: int = 2, blocks: int = 1, layers: int = 4):
        super(STNorm, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size

        self.num_nodes = self.input_vars
        self.out_dim = 1
        self.in_dim = 1

        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool

        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.mlps = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.cross_product = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.in_dim,
                                    out_channels=channels,
                                    kernel_size=(1, 1))

        receptive_field = 1
        self.dropout = nn.Dropout(0.2)

        self.dilation = []

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.dilation.append(new_dilation)
                if self.tnorm_bool:
                    self.tn.append(TNorm(self.num_nodes, channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(channels))
                self.filter_convs.append(nn.Conv2d(in_channels=num * channels,
                                                   out_channels=channels,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=num * channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, kernel_size),
                                                 dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=channels,
                                                     out_channels=channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, 1)))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1), bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=channels,
                                    out_channels=self.out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.temporal_projector = nn.Linear(self.input_window_size - blocks * (2 ** layers - 1),
                                            self.output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x -> [batch_size, input_window_size, input_size]"""
        x = x.unsqueeze(-1)  # -> [batch_size, input_window_size, input_size, 1]

        input = x.transpose(1, 3).contiguous()  # -> [batch_size, 1, input_size, input_window_size]
        in_len = input.size(3)  # -> input_window_size
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input  # -> [batch_size, 1, input_size, input_window_size]
        x = self.start_conv(x)  # -> [batch_size, channels(start_conv:out_channels), input_size, input_window_size]
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            x_list = []
            x_list.append(x)
            b, c, n, t = x.shape
            if self.tnorm_bool:
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
            if self.snorm_bool:
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)
            # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            b, c, n, t = filter.shape
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))  # -> [batch_size, channels, input_size, 衰减的input_window_size]
        out = self.end_conv_2(rep)  # -> [batch_size, out_dim(1), input_size, 衰减的input_window_size]
        out = self.temporal_projector(out)  # -> [batch_size, out_dim(1), input_size, output_window_size]
        out = out.squeeze(1)  # -> [batch_size, input_size, output_window_size]
        out = out.transpose(1, 2).contiguous()  # -> [batch_size, output_window_size, input_size]
        return out

    def load_my_state_dict(self, state_dict: dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)
