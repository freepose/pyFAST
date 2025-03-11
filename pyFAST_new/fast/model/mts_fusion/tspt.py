#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from ..mts import GAR

from ...data import InstanceScale, InstanceStandardScale


class TSPT(nn.Module):
    """
        Temporal Structure Preserving Transformer (TSPT)
        Author: Senzhen Wu, 20240308

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
        :param use_feu: whether to use feature enhance.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = 1,
                 num_layers: int = 3, num_heads: int = 4, d_model: int = 16, dim_ff: int = 128,
                 d_k: int = None, d_v: int = None, dropout_rate: float = 0.1,
                 use_instance_scale: bool = True,
                 use_feu: bool = True):
        super(TSPT, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.exogenous_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        self.embed = nn.Linear(self.ex_vars + 1, d_model)
        self.dropout = nn.Dropout(dropout_rate)

        W_pos = torch.empty((self.input_window_size + 1, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_k, d_v, dim_ff, dropout_rate
                          ) for _ in range(num_layers)])

        self.projection = GAR(d_model, 1)
        self.feu = FeatureEnhanceUnit(1.) if use_feu else nn.Identity()
        self.ar = GAR(input_window_size + 1, output_window_size)

        self.inst_scale = InstanceStandardScale() if use_instance_scale else InstanceScale()

    def forward(self, x: torch.Tensor, ex: torch.Tensor = None):
        x = self.inst_scale.fit_transform(x)

        # feature fusion unit
        if ex is not None:
            ex = ex.unsqueeze(3).tile(1, 1, 1,
                                      self.input_vars)  # [batch_size, input_window_size, ex_vars, input_vars] : Er
            x = x.unsqueeze(2)  # [batch_size, input_window_size, 1, input_vars] : Zr
            merge = torch.cat([ex, x], dim=2)  # [batch_size, input_window_size, ex_vars + 1, input_vars] : Zc
            window_mean = merge.mean(dim=1).unsqueeze(dim=1)  # [batch_size, ex_vars + 1, input_vars] : Zm
            x = torch.cat([merge, window_mean],
                          dim=1)  # [batch_size, input_window_size + 1, ex_vars + 1, input_vars] : Zw
            x = x.permute(0, 3, 1, 2)  # [batch_size, input_vars, input_window_size + 1, ex_vars + 1] : Zp
            x = x.flatten(0, 1)  # [batch_size * input_vars, input_window_size + 1, ex_vars + 1] : Zf
        else:
            merge = x.unsqueeze(2).tile(1, 1, self.ex_vars + 1,
                                        1)  # [batch_size, input_window_size, ex_vars + 1, input_vars]
            window_mean = merge.mean(dim=1).unsqueeze(dim=1)  # [batch_size, input_window_size, ex_vars + 1, input_vars]
            x = torch.cat([merge, window_mean], dim=1)  # [batch_size, input_window_size + 1, ex_vars + 1, input_vars]
            x = x.permute(0, 3, 1, 2)  # [batch_size, input_vars, input_window_size + 1, ex_vars + 1]
            x = x.flatten(0, 1)  # [batch_size * input_vars, input_window_size + 1, ex_vars + 1]

        # encoder
        x = self.embed(x)  # [batch_size * input_vars, input_window_size + 1, d_model] : L
        x = self.dropout(x)
        x = x + self.W_pos
        scores = None
        for atten_layer in self.layers:
            x, scores = atten_layer(x, prev=scores)
            scores = self.dropout(scores)

        # projection
        x = x.unflatten(0, (-1, self.input_vars))  # [batch_size, input_vars, input_window_size + 1, d_model]
        x = x.permute(0, 2, 3, 1)  # [batch_size, input_window_size + 1, d_model, input_vars]
        x = self.projection(x).squeeze(2)  # [batch_size, input_window_size + 1, input_vars]

        # feature enhance
        x = self.feu(x)

        # autoregression
        x = self.ar(x)  # [batch, output_window_size, input_size]

        out = self.inst_scale.inverse_transform(x)

        return out


class FeatureEnhanceUnit(nn.Module):
    """
        Feature enhance unit.
    """

    def __init__(self, p: float = 1.):
        super(FeatureEnhanceUnit, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        center_point = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, input_size]
        relative = x - center_point
        distance = torch.cdist(x, center_point, p=2)  # [batch_size, window_size, 1]
        mean_distance = torch.mean(distance, dim=1, keepdim=True)  # [batch_size, 1, 1]
        ratio_distance = distance / mean_distance  # [batch_size, window_size, 1]
        new_x = center_point + relative * (ratio_distance ** self.p)
        return new_x


class EncoderLayer(nn.Module):
    """
        Encoder layer of Transformer-like module.
    """

    def __init__(self, d_model: int, num_heads: int,
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

        # scaled dot-product attention
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
        q_s = self.W_Q(Q).view(bs, -1, self.num_heads, self.d_k).transpose(1,
                                                                           2)  # [batch_size, num_heads, window_size, d_k]
        k_s = self.W_K(K).view(bs, -1, self.num_heads, self.d_k).permute(0, 2, 3,
                                                                         1)  # [batch_size, num_heads, d_k, window_size]
        v_s = self.W_V(V).view(bs, -1, self.num_heads, self.d_v).transpose(1,
                                                                           2)  # [batch_size, num_heads, window_size, d_v]

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
