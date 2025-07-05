#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from typing import List
from ...data import InstanceStandardScale, PatchMaker
from ..base.decomposition import DecomposeSeries


class PatchEmbedding(nn.Module):
    """
        Patch embedding.

        :param seq_len: The length of the input sequence.
        :param patch_len: The length of each patch.
        :param patch_stride: The distances/steps of the patches.
        :param padding: The padding size for the input sequence.
        :param d_model: The dimension/hidden size of the model.
    """
    def __init__(self, seq_len: int, patch_len: int = 1, patch_stride: int = 1, padding: int = 0, d_model: int = 512):
        super(PatchEmbedding, self).__init__()

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.padding = padding
        self.d_model = d_model

        self.patch_maker = PatchMaker(seq_len, patch_len, patch_stride, padding)
        self.hidden_size = self.d_model // self.patch_maker.patch_num
        assert self.hidden_size > 0, "hidden_size must be greater than 0."

        self.l1 = nn.Linear(self.patch_len, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size * self.patch_maker.patch_num, self.d_model)

    def forward(self, x):
        """
            :param x: Input tensor of shape (batch_size, seq_len, input_vars)
        """
        x = self.patch_maker(x) # -> (batch_size, input_vars, patch_num, patch_len)
        x = self.l1(x) # -> (batch_size, input_vars, patch_num, hidden_size)
        x = x.flatten(start_dim=-2)  # -> (batch_size, input_vars, patch_num * hidden_size)
        x = self.l2(x)  # -> (batch_size, input_vars, d_model)

        return x


class MultiScalePatchEmbedding(nn.Module):
    """
        Multiscale patch embedding.

        :param seq_len: The length of the input sequence.
        :param patch_lens: List of patch lengths for different scales.
        :param d_model: The dimension of the model.
    """
    def __init__(self, seq_len: int, patch_lens: List[int] = [48, 24, 12, 6], d_model: int = 512):
        super(MultiScalePatchEmbedding, self).__init__()
        self.seq_len = seq_len
        self.patch_lens = patch_lens    # the patch lengths for different scales
        assert d_model % len(patch_lens) == 0, "d_model must be divisible by the number of patch lengths / scales."
        self.d_model = d_model // len(patch_lens)

        self.patch_embeddings = nn.ModuleList([
            PatchEmbedding(self.seq_len, patch_len, patch_len // 2, 0, self.d_model) for patch_len in self.patch_lens
        ])

    def forward(self, x):
        """
            :param x: Input tensor of shape (batch_size, seq_len, input_vars)
        """
        embeddings = [patch_embedding(x) for patch_embedding in self.patch_embeddings]
        x = torch.cat(embeddings, dim=-1)   # -> (batch_size, input_vars, d_model * len(patch_lens))

        return x


class Encoder(nn.Module):
    """
        Encoder block for the PatchMLP model. Maybe somewhat like a two-layer residual NN.

        :param d_model: The dimension of the model.
        :param n_vars: The number of input variables (features / channels).
    """
    def __init__(self, d_model: int, n_vars: int):
        super(Encoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(
            nn.Linear(n_vars, n_vars),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        """
            :param x: Input tensor of shape (batch_size, n_vars, d_model)
        """
        y_0 = self.ff1(x)   # -> (batch_size, n_vars, d_model)
        y_0 = y_0 + x       # -> (batch_size, n_vars, d_model)
        y_0 = self.norm1(y_0)

        y_1 = y_0.permute(0, 2, 1)  # -> (batch_size, d_model, n_vars)
        y_1 = self.ff2(y_1)         # -> (batch_size, d_model, n_vars)
        y_1 = y_1.permute(0, 2, 1)  # -> (batch_size, n_vars, d_model)

        y_2 = y_1 * y_0 + x         # -> (batch_size, n_vars, d_model)
        y_2 = self.norm1(y_2)

        return y_2


class PatchMLP(nn.Module):
    """
        Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting.
        Peiwang Tang, Weitai Zhang.
        The 39th AAAI Conference on Artificial Intelligence (AAAI-25).

        Author provided code: https://github.com/TangPeiwang/PatchMLP

        :param input_window_size: The length of the input sequence.
        :param input_vars: The number of input variables (features).
        :param output_window_size: The length of the output sequence.
        :param kernel_size: The kernel size for the decomposition.
        :param d_model: The dimension of the model.
        :param patch_lens: List of patch lengths for different scales.
        :param num_encoder_layers: The number of encoder layers.
        :param use_instance_scale: Whether to use instance standard scaling.
    """
    def __init__(self, input_window_size: int, input_vars: int = 1, output_window_size: int = 1,
                 kernel_size: int = 13, d_model: int = 512, patch_lens: List[int] = [48, 24, 12, 6],
                 num_encoder_layers: int = 1,
                 use_instance_scale: bool = True):
        super(PatchMLP, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.patch_lens = patch_lens
        self.num_encoder_layers = num_encoder_layers
        self.use_instance_scale = use_instance_scale

        self.decomposition = DecomposeSeries(kernel_size)
        self.patch_embeddings = MultiScalePatchEmbedding(self.input_window_size, self.patch_lens, self.d_model)

        self.seasonal_layers = nn.ModuleList([
            Encoder(self.d_model, self.input_vars) for _ in range(self.num_encoder_layers)
        ])
        self.trend_layers = nn.ModuleList([
            Encoder(self.d_model, self.input_vars) for _ in range(self.num_encoder_layers)
        ])

        self.projector = nn.Linear(self.d_model, self.output_window_size)

        if self.use_instance_scale:
            self.inst_scaler = InstanceStandardScale(self.input_vars, 1e-5)

    def forward(self, x):
        """
            :param x: Input tensor of shape (batch_size, input_window_size, input_vars)
        """

        if self.use_instance_scale:
            x = self.inst_scaler.fit_transform(x)

        x = self.patch_embeddings(x)    # -> (batch_size, input_vars, d_model)

        trend_init, seasonal_init = self.decomposition(x)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)
        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        x = seasonal_init + trend_init

        out = self.projector(x)     # -> (batch_size, input_vars, output_window_size)
        out = out.permute(0, 2, 1)  # -> (batch_size, output_window_size, input_vars)

        if self.use_instance_scale:
            out = self.inst_scaler.inverse_transform(out)

        return out
