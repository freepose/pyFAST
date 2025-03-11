#!/usr/bin/env python
# encoding: utf-8

import math
from typing import List, Literal

import torch
import torch.nn as nn

from .embedding import PositionalEncoding
from ...base import get_activation_cls
from ....data import PatchMaker, InstanceScale, InstanceStandardScale


class FullAttention(nn.Module):
    """
        Simple attention mechanism.
    """

    def __init__(self, dropout_rate: float = 0.1):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / math.sqrt(E)

        scores = torch.einsum("blhe, bshe -> bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls, bshd -> blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    """
        Simple attention wrapper layer, which implements multi-head.
    """

    def __init__(self, attention: nn.Module, d_model: int, num_heads: int, d_keys: int = None, d_values: int = None):
        super(AttentionLayer, self).__init__()
        self.inner_attention = attention
        self.num_heads = num_heads

        d_keys = d_keys or (d_model // num_heads)
        d_values = d_values or (d_model // num_heads)

        self.query_projection = nn.Linear(d_model, d_keys * num_heads)
        self.key_projection = nn.Linear(d_model, d_keys * num_heads)
        self.value_projection = nn.Linear(d_model, d_values * num_heads)
        self.out_projection = nn.Linear(d_values * num_heads, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.num_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values)
        out = out.view(B, L, -1)

        return self.out_projection(out)


class FlattenHead(nn.Module):
    def __init__(self, d_model: int, head_nf: int, output_window_size: int, dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_nf, output_window_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """ x -> (batch_size, input_vars, d_model, patch_num + 1) """
        x = self.flatten(x)  # -> (batch_size, input_vars, d_model * (patch_num + 1))
        x = self.linear(x)  # -> (batch_size, input_vars, output_window_size)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, input_window_size: int, input_vars: int, d_model: int, patch_len: int, dropout_rate: float):
        super(EnEmbedding, self).__init__()
        self.glb_token = nn.Parameter(torch.randn(1, input_vars, 1, d_model))
        self.patch_maker = PatchMaker(input_window_size, patch_len, patch_len)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.pe = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size, input_vars] """
        batch_size, input_window_size, input_vars = x.shape

        # do patching
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))  # -> (batch_size, input_vars, 1, d_model)

        x = self.patch_maker(x)  # -> (batch_size, input_vars, patch_num, patch_len)
        x = x.flatten(0, 1)  # -> (batch_size * input_vars, patch_num, patch_len)

        # Input encoding
        x = self.value_embedding(x) + self.pe(x)  # -> (batch_size * input_vars, patch_num, d_model)
        x = x.unflatten(0, (batch_size, input_vars))  # -> (batch_size, input_vars, patch_num, d_model)
        x = torch.cat([x, glb], dim=2)  # -> (batch_size, input_vars, patch_num + 1, d_model)
        x = x.flatten(0, 1)  # -> (batch_size * input_vars, patch_num + 1, d_model)
        x = self.dropout(x)

        return x


class Encoder(nn.Module):
    def __init__(self, layers: List[nn.Module], norm_layer: nn.Module):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        for layer in self.layers:
            x = layer(x, cross)
        x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module,
                 d_model: int, dim_ff: int, dropout_rate: float, activation: str):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=dim_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=dim_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        """
            x -> (batch_size * input_vars, patch_num + 1, d_model)
            cross -> (batch_size, input_window_size, d_model)
        """
        batch_size, input_window_size, d_model = cross.shape

        x = x + self.dropout(self.self_attention(x, x, x))  # -> (batch_size * input_vars, patch_num + 1, d_model)
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)  # -> (batch_size * input_vars, 1, d_model)
        x_glb = x_glb_ori.reshape(batch_size, -1, d_model)  # -> (batch_size, input_vars, d_model)
        x_glb_attn = self.dropout(self.cross_attention(x_glb, cross, cross))  # -> (batch_size, input_vars, d_model)
        x_glb_attn = x_glb_attn.flatten(0, 1)  # -> (batch_size * input_vars, d_model)
        x_glb_attn = x_glb_attn.unsqueeze(1)  # -> (batch_size * input_vars, 1, d_model)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # -> (batch_size * input_vars, patch_num + 1, d_model)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class TimeXer(nn.Module):
    """
        Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, Mingsheng Long
        TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables
        NIPS 2024. url: https://arxiv.org/abs/2402.19072

        Official Code: https://github.com/thuml/TimeXer
        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/TimeXer.py (our implementation)

        Encoder-only transformer.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
        :param activation: activation unit function name.
        :param patch_len: patch length.
        :param use_instance_scale: whether to use instance standard scale (a.k.a., RevIN).
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 d_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 2,
                 dim_ff: int = 2048, dropout_rate: float = 0.05,
                 activation: Literal['relu', 'gelu'] = 'relu',
                 patch_len: int = 16, use_instance_scale: bool = True):
        super(TimeXer, self).__init__()
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

        # en_embedding
        self.en_embedding = EnEmbedding(input_window_size, input_vars, d_model, patch_len, dropout_rate)

        # ex_embedding
        self.value_embedding = nn.Linear(input_vars, d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # encoder
        self_attention = FullAttention(dropout_rate)
        corss_attention = FullAttention(dropout_rate)
        self_attention_layer = AttentionLayer(self_attention, d_model, num_heads)
        corss_attention_layer = AttentionLayer(corss_attention, d_model, num_heads)
        encoder_layer = EncoderLayer(self_attention_layer, corss_attention_layer, d_model, dim_ff, dropout_rate,
                                     activation)
        encoder_layers = [encoder_layer for _ in range(num_encoder_layers)]
        norm_layer = nn.LayerNorm(d_model)
        self.encoder = Encoder(encoder_layers, norm_layer)

        self.head = FlattenHead(d_model, d_model * (self.en_embedding.patch_maker.patch_num + 1),
                                output_window_size, dropout_rate)

        self.inst_scale = InstanceStandardScale() if use_instance_scale else InstanceScale()

    def forward(self, x: torch.Tensor):
        """
            x -> [batch_size, input_window_size, input_size]
            we adapt ``forecast_multi`` in TS-Library
        """

        norm_x = self.inst_scale.fit_transform(x)

        en_embedding = self.en_embedding(norm_x)  # -> (batch_size * input_vars, patch_num + 1, d_model)
        ex_embedding = self.value_embedding(norm_x)  # -> (batch_size, input_window_size, d_model)
        ex_embedding = self.embedding_dropout(ex_embedding)

        encoder_out = self.encoder(en_embedding, ex_embedding)  # -> (batch_size * input_vars, patch_num + 1, d_model)
        encoder_out = encoder_out.unflatten(0, (
        -1, self.input_vars))  # -> (batch_size, input_vars, patch_num + 1, d_model)
        encoder_out = encoder_out.transpose(-1, -2)  # -> (batch_size, input_vars, d_model, patch_num + 1)

        out = self.head(encoder_out)  # -> (batch_size, input_vars, output_window_size)
        out = out.transpose(1, 2)  # -> (batch_size, output_window_size, input_vars)

        out = self.inst_scale.inverse_transform(out)

        return out