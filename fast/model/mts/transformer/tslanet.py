#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from typing import Literal
from torch.nn.init import trunc_normal_

from fast.data import InstanceStandardScale, PatchMaker


class InteractiveConvBlock(nn.Module):
    """
        Interactive Convolution Block (ICB), applies convolutional operations to enhance feature extraction.

        :param input_vars: Number of input variables.
        :param hidden_size: Size of the hidden layer.
        :param dropout_rate: Dropout rate.
    """

    def __init__(self, input_vars: int = 1, hidden_size: int = 1, dropout_rate: float = 0.):
        super(InteractiveConvBlock, self).__init__()

        self.input_vars = input_vars
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv1d(self.input_vars, self.hidden_size, 1)
        self.conv2 = nn.Conv1d(self.input_vars, self.hidden_size, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(self.hidden_size, self.input_vars, 1)
        self.drop = nn.Dropout(p=self.dropout_rate)
        self.act = nn.GELU()

    def forward(self, x):
        """
            :param x: Input tensor with shape [batch_size, input_vars, sequence_length]
        """
        x = x.transpose(1, 2)  # -> [batch_size, sequence_length, input_vars]
        x1 = self.conv1(x)  # -> [batch_size, sequence_length, hidden_size]
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)  # -> [batch_size, sequence_length, hidden_size]
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class AdaptiveSpectralBlock(nn.Module):
    def __init__(self, embedding_dim: int, adaptive_filter: bool = True):
        super(AdaptiveSpectralBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.adaptive_filter = adaptive_filter

        # weight for high frequencies and original frequencies
        self.complex_weight_high = nn.Parameter(torch.randn(self.embedding_dim, 2) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(self.embedding_dim, 2) * 0.02)

        # Initialize weights with truncated normal distribution (mean=0, std=0.02)
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)

        # Learnable threshold parameter: decide which frequencies to keep
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)    # maybe not need 0.5

    def create_adaptive_high_freq_mask(self, x_fft):
        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = torch.flatten(energy, start_dim=1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(-1, 1)  # Reshape to match the original dimensions
        normalized_energy = energy / (median_energy + 1e-6)  # median normalization

        # Create a hard mask based on the threshold parameter
        hard_mask = (normalized_energy > self.threshold_param).float()

        # Ensure the mask is differentiable
        adaptive_mask = hard_mask.detach() + self.threshold_param - self.threshold_param.detach()

        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x):
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        # Convert complex weights once
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Combine mask creation and high frequency processing
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted += x_fft * freq_mask * weight_high

        out = torch.fft.irfft(x_weighted, n=x.size(1), dim=1, norm='ortho')
        out = out.view_as(x)

        return out


class TSLABlock(nn.Module):
    """
        TSLA Block, combining Adaptive Spectral Block and ICB model.

        :param embedding_dim: Embedding dimension.
        :param hidden_size: Hidden size for ICB, defaults to embedding_dim * 3.
        :param block_type: Type of block to use, can be 'asb', 'icb', or 'asb_icb'.
        :param dropout_rate: Dropout rate.
    """

    def __init__(self, embedding_dim: int, hidden_size: int = None,
                 block_type: Literal['asb', 'icb', 'asb_icb'] = 'asb_icb',
                 dropout_rate: float = 0.):
        super(TSLABlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size if hidden_size is not None else embedding_dim * 3
        self.block_type = block_type
        self.dropout_rate = dropout_rate

        if self.block_type == 'asb':
            self.block = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                AdaptiveSpectralBlock(self.embedding_dim),
                nn.Dropout(self.dropout_rate)
            )
        elif self.block_type == 'icb':
            self.block = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                InteractiveConvBlock(self.embedding_dim, self.hidden_size, self.dropout_rate),
                nn.Dropout(self.dropout_rate)
            )
        elif self.block_type == 'asb_icb':
            self.block = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                AdaptiveSpectralBlock(self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                InteractiveConvBlock(embedding_dim, self.hidden_size, self.dropout_rate),
                nn.Dropout(self.dropout_rate)
            )

    def forward(self, x):
        """
            :param x: Input tensor with shape [batch_size, patch_num, embedding_dim]
        """
        x = x + self.block(x)

        return x


class TSLANet(nn.Module):
    """
        TSLANet: Rethinking Transformers for Time Series Representation Learning.
        Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Li, Xiaoli.
        url: https://arxiv.org/pdf/2404.08472

        Author provide code: https://github.com/emadeldeen24/TSLANet

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param patch_len: length of each patch.
        :param patch_stride: stride of the patch, defaults to patch_len // 2.
        :param embedding_dim: dimension of the embedding layer.
        :param mlp_hidden_size: hidden size for MLP, defaults to embedding_dim * 3.
        :param num_blocks: number of TSLABlocks.
        :param block_type: type of block to use, can be 'asb', 'icb', or 'asb_icb'.
        :param dropout_rate: dropout rate for the blocks.
        :param use_instance_scale: whether to use instance standard scaling, defaults to True.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 patch_len: int = 16, patch_stride: int = None,
                 embedding_dim: int = 32, mlp_hidden_size: int = None,
                 num_blocks: int = 1, block_type: Literal['asb', 'icb', 'asb_icb'] = 'asb_icb',
                 dropout_rate: float = 0., use_instance_scale: bool = True):
        super(TSLANet, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size

        self.patch_len = patch_len
        self.patch_stride = patch_stride if patch_stride is not None else self.patch_len // 2

        self.embedding_dim = embedding_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.num_blocks = num_blocks
        self.block_type = block_type

        self.dropout_rate = dropout_rate
        self.use_instance_scale = use_instance_scale

        self.patch_maker = PatchMaker(self.input_window_size, self.patch_len, self.patch_stride, 0)

        # Data Embedding Layer
        self.l1 = nn.Linear(self.patch_len, self.embedding_dim)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.dropout_rate, self.num_blocks)]

        self.blocks = nn.ModuleList([
            TSLABlock(self.embedding_dim, self.mlp_hidden_size, block_type=self.block_type, dropout_rate=dpr[i])
            for i in range(self.num_blocks)]
        )

        self.l2 = nn.Linear(self.embedding_dim * self.patch_maker.patch_num, self.output_window_size)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(-1, 1e-5)

    def forward(self, x):
        """
            :param x: Input tensor with shape [batch_size, input_window_size, input_vars]
        """
        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x)

        patches = self.patch_maker(x)  # -> [batch_size, input_vars, patch_num, patch_len]

        x = patches.flatten(0, 1)  # -> [batch_size * input_vars, patch_num, patch_len]
        x = self.l1(x)  # -> [batch_size * input_vars, patch_num, embedding_dim]

        for block in self.blocks:
            x = block(x)  # -> [batch_size * input_vars, patch_num, embedding_dim]

        x = x.flatten(1, 2)  # -> [batch_size * input_vars, patch_num * embedding_dim]

        out = self.l2(x)  # -> [batch_size * input_vars, output_window_size]
        out = out.reshape(-1, self.input_vars, self.output_window_size).transpose(1, 2)  # -> [B, output_window_size, input_vars]

        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        return out
