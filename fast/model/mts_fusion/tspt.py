#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from ...data import InstanceScale, InstanceStandardScale, PatchMaker
from ..mts import GAR


class TSPT(nn.Module):
    """
        Temporal Structure Preserving Transformer (TSPT).

        Author: Senzhen Wu, 20240315

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param num_layers: number of attention layers.
        :param num_heads: head number, a.k.a., attention number.
        :param d_model: model dimension, a.k.a., embedding size.
        :param dim_ff: feed forward dimension.
        :param d_k: dimension of key matrix in attention module.
        :param d_v: dimension of value matrix in attention module.
        :param dropout_rate: dropout rate.
        :param use_instance_scale: whether to use instance standard scale. InstanceStandardScale is a.k.a. RevIN.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 ex_vars: int = 1, ex_linear_layers: list = [32], target_linear_layers: list = [32],
                 variable_hidden_size: int = 16, patch_len: int = 24, patch_stride: int = 24, patch_padding: int = 0,
                 num_layers: int = 3, num_heads: int = 4, d_model: int = 64, dim_ff: int = 128,
                 d_k: int = None, d_v: int = None, dropout_rate: float = 0.1,
                 use_instance_scale: bool = True):
        super(TSPT, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.ex_vars = ex_vars
        self.variable_hidden_size = variable_hidden_size

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding

        ex_linear_hidden_sizes = [ex_vars] + ex_linear_layers + [variable_hidden_size]
        self.ex_linear_layers = []
        for i in range(len(ex_linear_hidden_sizes) - 1):
            self.ex_linear_layers.append(nn.Linear(ex_linear_hidden_sizes[i], ex_linear_hidden_sizes[i + 1]))
        self.ex_linear_layers = nn.ModuleList(self.ex_linear_layers)

        target_linear_hidden_sizes = [input_vars, ] + target_linear_layers + [variable_hidden_size, ]
        self.target_linear_layers = []
        for i in range(len(target_linear_hidden_sizes) - 1):
            self.target_linear_layers.append(
                nn.Linear(target_linear_hidden_sizes[i], target_linear_hidden_sizes[i + 1]))
        self.target_linear_layers = nn.ModuleList(self.target_linear_layers)

        self.patch_maker = PatchMaker(input_window_size, patch_len, patch_stride, patch_padding)

        self.weight = nn.Parameter(torch.ones(variable_hidden_size, self.patch_maker.patch_num, patch_len), )
        self.embed = nn.Linear(patch_len, d_model)

        self.dropout = nn.Dropout(dropout_rate)

        W_pos = torch.empty((self.patch_maker.patch_num, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        self.layers = nn.ModuleList(
            [EncoderLayer(self.patch_maker.patch_num, d_model, num_heads, d_k, d_v, dim_ff, dropout_rate
                          ) for _ in range(num_layers)])

        self.autoregression = GAR(self.patch_maker.patch_num * d_model, output_window_size)
        self.mapping = nn.Linear(variable_hidden_size, output_vars)

        self.inst_scale = InstanceStandardScale() if use_instance_scale else InstanceScale()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, ex: torch.Tensor = None):
        """
            :param x: shape is (batch_size, input_window_size, input_vars)
            :param ex: shape is (batch_size, input_window_size, ex_vars)
            :return: shape is (batch_size, output_window_size, output_vars)
        """
        x = self.inst_scale.fit_transform(x)

        out_ex = ex.clone()
        for lin in self.ex_linear_layers:
            out_ex = lin(out_ex)
        # [batch_size, input_window_size, variable_hidden_size]

        out_x = x.clone()
        for lin in self.target_linear_layers:
            out_x = lin(out_x)
        # [batch_size, input_window_size, variable_hidden_size]

        out_ex = self.patch_maker(out_ex)  # [batch_size, variable_hidden_size, patch_num, patch_len]
        out_x = self.patch_maker(out_x)  # [batch_size, variable_hidden_size, patch_num, patch_len]

        out_x = out_x * out_ex * self.weight

        # encoder
        out_x = self.embed(out_x)  # [batch_size, variable_hidden_size, patch_num, d_model]
        out_x = out_x.flatten(0, 1)  # [batch_size * variable_hidden_size, patch_num, d_model]
        out_x = self.dropout(out_x)
        out_x = out_x + self.W_pos
        out_x = self.dropout(out_x)
        scores = None
        for atten_layer in self.layers:
            out_x, scores = atten_layer(out_x, prev=scores)
            scores = self.dropout(scores)

        # projection
        out_x = out_x.unflatten(0, (
            -1, self.variable_hidden_size))  # [batch_size, variable_hidden_size, patch_num, d_model]
        out_x = out_x.flatten(2, 3)  # [batch_size, variable_hidden_size, patch_num * d_model]
        out_x = out_x.transpose(1, 2)  # [batch_size, patch_num * d_model, variable_hidden_size]
        out_x = self.autoregression(out_x)  # [batch_size, output_window_size, variable_hidden_size]
        out_x = self.mapping(out_x)  # [batch_size, output_window_size, output_vars]

        out = self.inst_scale.inverse_transform(out_x)
        return out


class EncoderLayer(nn.Module):
    """
        Encoder layer of Transformer-like module.
    """

    def __init__(self, dim1_size: int, d_model: int, num_heads: int,
                 d_k: int = None, d_v: int = None, dim_ff: int = 256, dropout_rate: float = 0., bias=True):
        super(EncoderLayer, self).__init__()
        d_k = d_model // num_heads if d_k is None else d_k
        d_v = d_model // num_heads if d_v is None else d_v

        # multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, d_k, d_v, dropout_rate=dropout_rate)
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff, bias=bias),
                                nn.GELU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(dim_ff, d_model, bias=bias))
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, src: torch.Tensor, prev: torch.Tensor = None):
        src2, scores = self.self_attn(src, src, src, prev)  # [batch_size, window_size, d_model]
        src = src + self.dropout_attn(src2)  # [batch_size, window_size, d_model]
        src2 = self.ff(src2)  # [batch_size, window_size, d_model]
        # src = src + self.dropout_attn(src1)  # [batch_size, window_size, d_model]
        # src2 = self.ff(src1)  # [batch_size, window_size, d_model]
        src = src + self.dropout_ffn(src2)  # [batch_size, window_size, d_model]
        return src, scores


class MultiHeadAttention(nn.Module):
    """
        Multi-head attention module using scaled dot product attention.
    """

    def __init__(self, d_model: int, num_heads: int, d_k: int, d_v: int, dropout_rate: float = 0., bias: bool = True):
        super(MultiHeadAttention, self).__init__()
        self.num_heads, self.d_k, self.d_v = num_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * num_heads, bias=bias)
        self.W_K = nn.Linear(d_model, d_k * num_heads, bias=bias)
        self.W_V = nn.Linear(d_model, d_v * num_heads, bias=bias)

        self.sdp_attn = ScaledDotProductAttention(d_model, num_heads, dropout_rate=dropout_rate)

        # projection
        self.to_out = nn.Sequential(
            nn.Linear(num_heads * d_v, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, prev: torch.Tensor = None):
        """ Q, K, V: [batch_size, window_size, d_model] """
        bs = Q.size(0)

        # linear and split in multiple heads
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)  # [bs, nheads, window_size, d_k]
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.d_k).permute(0, 2, 3, 1)  # [bs, nheads, d_k, window_size]
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.d_v).transpose(1, 2)  # [bs, nheads, window_size, d_v]

        # apply scaled dot-product attention (multiple heads)
        output, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev)
        # output: [batch_size, num_heads, window_size, d_v]
        # attn_scores: [batch_size, num_heads, window_size, window_size]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous()  # [batch_size, window_size, num_heads, d_v]
        output = output.view(bs, -1, self.num_heads * self.d_v)  # [batch_size, window_size, num_heads * d_v]
        output = self.to_out(output)  # [batch_size, window_size, d_model]

        return output, attn_scores


class ScaledDotProductAttention(nn.Module):
    """
        Scaled dot-product attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        head_dim = d_model // num_heads if d_model // num_heads > 1. else 1.
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prev: torch.Tensor = None):
        """
            :param q: [batch_size, num_heads, window_size, d_k]
            :param k: [batch_size, num_heads, d_k, window_size]
            :param v: [batch_size, num_heads, window_size, d_v]
            :param prev: [batch_size, num_heads, window_size, window_size]
        """
        attn_scores = torch.matmul(q, k) * self.scale  # [batch_size, num_heads, window_size, window_size]
        if prev is not None:
            attn_scores = attn_scores + prev
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, window_size, window_size]
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, window_size, d_v]
        return output, attn_scores
