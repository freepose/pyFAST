#!/usr/bin/env python
# encoding: utf-8

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import PositionalEncoding
from ....data import InstanceScale, InstanceStandardScale


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
        Encoder Layer for VanillaTransformer.
    """

    def __init__(self, scale: float = None, dropout_rate: float = 0.1, mask_flag: bool = True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model=512, n_heads=4, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        # x: [batch_size, window_size, d_model]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # => [batch_size, window_size, n_heads, d_model]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)  # => [batch_size, window_size, n_heads * d_model]

        return self.out_projection(out), attn  # => [batch_size, window_size, d_model]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout, position_embedding=True):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.positioned = position_embedding

        # Positional embedding
        if position_embedding:
            self.position_embedding = PositionalEncoding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]  # [B, M, T]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, M, N, L]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        if self.positioned:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x)
        return self.dropout(x), n_vars


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class TimerBackbone(nn.Module):
    def __init__(self, patch_len, d_model, n_heads, e_layers, d_ff, activation, dropout_rate):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.layers = e_layers
        self.d_ff = d_ff

        padding = 0

        # patching and embedding
        self.patch_embedding = PatchEmbedding(self.d_model, self.patch_len, self.stride, padding, dropout_rate)

        # Decoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(dropout_rate=dropout_rate, mask_flag=True),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=dropout_rate,
                    activation=activation
                ) for _ in range(self.layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Prediction Head
        self.proj = nn.Linear(self.d_model, self.patch_len, bias=True)


class Timer(nn.Module):
    """
        Liu, Yong and Zhang, Haoran and Li, Chenyu and Huang, Xiangdong and Wang, Jianmin and Long, Mingsheng
        Timer: Generative Pre-trained Transformers Are Large Time Series Models, ICML 2024.
        url: https://arxiv.org/pdf/2402.02368
        Official Code: https://github.com/thuml/Large-Time-Series-Model

        Encoder-only Transformer. Vary length input window.
     """

    def __init__(self, output_window_size: int, patch_len: int = 24,
                 d_model: int = 512, num_heads: int = 8, e_layers: int = 2, dim_ff: int = 2048,
                 activation: str = 'gelu', dropout_rate: float = 0.1):
        super().__init__()
        self.output_window_size = output_window_size
        self.patch_len = patch_len
        self.stride = patch_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.layers = e_layers
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate

        self.backbone = TimerBackbone(patch_len, d_model, num_heads, e_layers, dim_ff, activation, dropout_rate)

        # Decoder
        self.decoder = self.backbone.decoder
        self.proj = self.backbone.proj

        self.enc_embedding = self.backbone.patch_embedding

        self.inst_scale = InstanceStandardScale()

    def forward(self, x_enc):
        """
            将forecast改名为forward
        """
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        x_enc = self.inst_scale.fit_transform(x_enc)

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # [B, M, T]
        dec_in, n_vars = self.enc_embedding(x_enc)  # [B * M, N, D]

        # Encoder
        dec_out, _ = self.decoder(dec_in)  # [B * M, N, D]
        dec_out = self.proj(dec_out)  # [B * M, N, L]
        dec_out = dec_out.reshape(B, M, -1).transpose(1, 2)  # [B, T, M]

        # De-Normalization from Non-stationary Transformer
        dec_out = self.inst_scale.inverse_transform(dec_out)

        dec_out = dec_out[:, -self.output_window_size:, :]

        return dec_out

    # def encoder_top(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    #     x_enc /= stdev
    #
    #     # do patching and embedding
    #     x_enc = x_enc.permute(0, 2, 1)
    #     # u: [bs * nvars x patch_num x d_model]
    #     dec_in, n_vars = self.enc_embedding(x_enc)
    #
    #     # Encoder
    #     # z: [bs * nvars x patch_num x d_model]
    #
    #     return dec_in
    #
    # def encoder_bottom(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
    #     # Normalization from Non-stationary Transformer
    #     means = x_enc.mean(1, keepdim=True).detach()
    #     x_enc = x_enc - means
    #     stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    #     x_enc /= stdev
    #
    #     # do patching and embedding
    #     x_enc = x_enc.permute(0, 2, 1)
    #     # u: [bs * nvars x patch_num x d_model]
    #     dec_in, n_vars = self.enc_embedding(x_enc)  # [B * M, N, D]
    #
    #     # Encoder
    #     dec_out, attns = self.decoder(dec_in)  # [B * M, N, D]
    #     return dec_out
