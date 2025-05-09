#!/usr/bin/env python
# encoding: utf-8

from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from .embedding import PositionalEncoding
from ....data import PatchMaker, InstanceScale, InstanceStandardScale
from ...base import get_activation_cls


class TriangularCausalMask:
    def __init__(self, B: int, L: int, device: str or torch.device):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
        Attention mechanism.
        :param mask_flag: use mask.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, mask_flag: bool = True, dropout_rate: float = 0.1):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: torch.Tensor):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), None


class AttentionLayer(nn.Module):
    """
        Attention wrapper layer, which implements multi-head.
    """

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int, d_keys: int = None, d_values: int = None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: torch.Tensor):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """
        Encoder layer for Transformer series.
        :param attention: the type of attention mechanism used.
        :param d_model: model dimension, a.k.a., embedding size.
        :param d_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
        :param activation: activation function.
    """

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None,
                 dropout_rate: float = 0.1, activation: str = "relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
        Encoder for Transformer series.
        :param attn_layers: attention layer.
        :param norm_layer: normalization layer.
    """

    def __init__(self, attn_layers: list, norm_layer: nn.Module):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class Transpose(nn.Module):
    """
        Transpose module.
    """

    def __init__(self, *dims, contiguous: bool = False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x: torch.Tensor):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class PatchTST(nn.Module):
    """
        Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam
        A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, ICLR 2023
        Link: https://arxiv.org/abs/2211.14730
        Official Code: https://github.com/yuqinie98/PatchTST

        This code is based on TS-Library.
        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/PatchTST.py

        Encoder-only transformer.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param dim_ff: feed forward dimension.
        :param activation: activation unit function name.
        :param dropout_rate: dropout rate.
        :param patch_len: patch length.
        :param patch_stride: patch stride.
        :param use_instance_scale: whether to use instance standard scale (a.k.a., RevIN).
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 2,
                 dim_ff: int = 2048,
                 activation: str = 'gelu',
                 dropout_rate: float = 0.05,
                 patch_len: int = 4,
                 patch_stride: int = 1,
                 patch_padding: int = 1,
                 use_instance_scale: bool = False):
        super().__init__()
        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding

        self.use_instance_scale = use_instance_scale

        self.patch_maker = PatchMaker(self.input_window_size, self.patch_len, self.patch_stride, self.patch_padding)
        self.value_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.pe = PositionalEncoding(self.d_model)

        # encoder
        self.encoder_attention = FullAttention(mask_flag=False, dropout_rate=self.dropout_rate)
        self.encoder_attention_layer = AttentionLayer(self.encoder_attention, self.d_model, self.num_heads)
        self.encoder_layer = EncoderLayer(self.encoder_attention_layer, self.d_model, self.dim_ff,
                                          self.dropout_rate, self.activation)
        self.encoder_layers = [self.encoder_layer for _ in range(self.num_encoder_layers)]

        self.encoder_norm_layer = nn.Sequential(Transpose(1, 2),
                                                nn.BatchNorm1d(self.d_model),
                                                Transpose(1, 2))
        self.encoder = Encoder(attn_layers=self.encoder_layers, norm_layer=self.encoder_norm_layer)

        self.linear1 = nn.Linear(self.d_model * self.patch_maker.patch_num, self.output_window_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.inst_scale = InstanceStandardScale() if use_instance_scale else InstanceScale()

    def forward(self, x: torch.Tensor):
        """ x -> (batch_size, input_window_size, input_vars) """

        norm_x = self.inst_scale.fit_transform(x)

        batch_size, input_window_size, input_vars = x.shape
        x_patches = self.patch_maker(norm_x)  # -> (batch_size, input_vars, patch_num, patch_len)
        x_patches = x_patches.flatten(0, 1)  # -> (batch_size * input_vars, patch_num, patch_len)

        patch_embedding = self.value_embedding(x_patches) + self.pe(x_patches)  # patch_len -> d_model

        encoder_out, _ = self.encoder(patch_embedding)  # -> (batch_size * input_vars, patch_num, d_model)

        encoder_out = encoder_out.unflatten(0, (batch_size, input_vars))
        encoder_out = encoder_out.permute(0, 1, 3, 2)  # -> (batch_size, input_vars, d_model, patch_num)
        encoder_out = encoder_out.flatten(2, 3)  # -> (batch_size, input_vars, d_model * patch_num)

        out = self.linear1(encoder_out)  # -> (batch_size, input_vars, output_window_size)
        out = self.dropout1(out)
        out = out.permute(0, 2, 1)  # -> (batch_size, output_window_size, input_vars)

        out = self.inst_scale.inverse_transform(out)

        return out
