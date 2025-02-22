#!/usr/bin/env python
# coding=utf-8

import math, typing
import torch
import torch.nn as nn

from .embedding import PositionalEncoding
from ....data import PatchMaker


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


class TwoStageAttentionLayer(nn.Module):
    """
        The Two Stage Attention (TSA) Layer
        input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    """

    def __init__(self, seg_num: int, factor: int, d_model: int, num_heads: int, dim_ff: int, dropout_rate: float):
        super(TwoStageAttentionLayer, self).__init__()
        self.time_attention = AttentionLayer(FullAttention(dropout_rate), d_model, num_heads)
        self.dim_sender = AttentionLayer(FullAttention(dropout_rate), d_model, num_heads)
        self.dim_receiver = AttentionLayer(FullAttention(dropout_rate), d_model, num_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, dim_ff), nn.GELU(), nn.Linear(dim_ff, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cross Time Stage: Directly apply MSA to each dimension

        time_in = x.flatten(0, 1)  # -> (batch_size * ts_d, seg_num, d_model)
        time_enc = self.time_attention(time_in, time_in, time_in)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)  # -> (batch_size * ts_d, seg_num, d_model)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and
        #                        distribute messages to build the D-to-D connection
        batch = x.shape[0]

        dim_send = dim_in.unflatten(0, (batch, -1))  # -> (batch_size, ts_d, seg_num, d_model)
        dim_send = dim_send.permute(0, 2, 1, 3)  # -> (batch_size, seg_num, ts_d, d_model)
        dim_send = dim_send.flatten(0, 1)  # -> (batch_size * seg_num, ts_d, d_model)

        batch_router = self.router.unsqueeze(0).expand(batch, -1, -1, -1).flatten(0, 1)

        dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
        dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = dim_enc.unflatten(0, (batch, -1)).transpose(1, 2)  # -> (batch_size, ts_d, seg_num, d_model)

        return final_out


class SegMerging(nn.Module):
    """
        Segment merging layer for Crossformer.
    """

    def __init__(self, d_model: int, win_size: int, norm_layer: typing.Type[nn.Module] = nn.LayerNorm):
        super(SegMerging, self).__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size

        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class ScaleBlock(nn.Module):
    """
    Scale block for Crossformer.
    """

    def __init__(self, win_size: int, d_model: int, num_heads: int, dim_ff: int, depth: int,
                 dropout_rate: float, seg_num: int,
                 factor: int):
        super(ScaleBlock, self).__init__()

        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()
        for _ in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, num_heads, dim_ff, dropout_rate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """
    Encoder for Crossformer.
    """

    def __init__(self, attn_layers: typing.List[ScaleBlock]):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        encode_x = [x]
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)
        return encode_x


class DecoderLayer(nn.Module):
    """
    Decoder layer for Crossformer.
    """

    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, seg_len: int, d_model: int,
                 dropout_rate: float):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        x = self.self_attention(x)

        x = x.flatten(0, 1)  # -> (batch_size * ts_d, out_seg_num, d_model)
        cross = cross.flatten(0, 1)  # -> (batch_size * ts_d, in_seg_num, d_model)

        tmp = self.cross_attention(x, cross, cross)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)  # -> (batch_size * ts_d, out_seg_num, d_model)

        dec_output = dec_output.unflatten(0, (batch, -1))  # -> (batch_size, ts_d, seg_num, d_model)
        layer_predict = self.linear_pred(dec_output)  # -> (batch_size, ts_d, seg_num, seg_len)
        layer_predict = layer_predict.flatten(1, 2)  # -> (batch_size, seg_num, seg_len)

        return dec_output, layer_predict


class Decoder(nn.Module):
    """
    Decoder for Crossformer.
    """

    def __init__(self, layers: typing.List[DecoderLayer]):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, cross: typing.List[torch.Tensor]) -> torch.Tensor:
        """
            :param x:  shape is (batch_size, ts_d, seg_num, d_model)
            :param cross: shape is [(batch_size, ts_d, in_seg_num, d_model), ...]
        """
        final_predict = None
        batch_size, ts_d, seg_num, d_model = x.shape

        for i, layer in enumerate(self.decode_layers):
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict

        final_predict = final_predict.unflatten(1, (-1, seg_num))  # -> (batch_size, ts_d, seg_num, seg_len)
        final_predict = final_predict.flatten(2, 3)  # -> (batch_size, ts_d, seg_num * seg_len)
        final_predict = final_predict.transpose(1, 2)  # -> (batch_size, seg_num * seg_len, ts_d)

        return final_predict


class Crossformer(nn.Module):
    """
        Yunhao Zhang, Junchi Yan
        Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.
        ICLR 2023. url: https://openreview.net/pdf?id=vSVLM2j9eie

            Official Code: https://github.com/Thinklab-SJTU/Crossformer/blob/master/cross_models/cross_former.py
        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/Crossformer.py

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param num_decoder_layers: number of decoder layers.
        :param dim_ff: feed forward dimension.
        :param factor: a scaling factor for ``ScaleBlock``.
        :param dropout_rate: dropout rate.
        :param seg_len: the length of each time series segment.
        :param win_size: the number of time periods included in each window.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 d_model: int = 512, num_heads: int = 8, num_encoder_layers: int = 2, num_decoder_layers: int = 1,
                 dim_ff: int = 2048, factor: int = 3, dropout_rate: float = 0.05,
                 seg_len: int = 12, win_size: int = 2):
        super(Crossformer, self).__init__()
        assert output_vars == input_vars, 'Invalid window parameters.'
        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.factor = factor
        self.dropout_rate = dropout_rate
        self.seg_len = seg_len
        self.win_size = win_size

        # The padding operation to handle invisible segment length
        self.pad_in_len = math.ceil(1.0 * self.input_window_size / self.seg_len) * self.seg_len
        self.pad_out_len = math.ceil(1.0 * self.output_window_size / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = math.ceil(self.in_seg_num / (self.win_size ** (self.num_encoder_layers - 1)))
        self.head_nf = self.d_model * self.out_seg_num

        # encoder embedding
        self.patch_maker = PatchMaker(self.input_window_size, self.seg_len, self.seg_len,
                                      self.pad_in_len - self.input_window_size)

        self.value_embedding = nn.Linear(self.seg_len, self.d_model, bias=False)
        self.encoder_pe = PositionalEncoding(self.d_model)
        self.encoder_dropout = nn.Dropout(self.dropout_rate)
        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, self.input_vars, self.in_seg_num, self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # decoder embedding
        self.decoder_pos_embedding = nn.Parameter(
            torch.randn(1, self.input_vars, (self.pad_out_len // self.seg_len), self.d_model))

        # encoder
        self.encoder_scale_blocks = []
        self.encoder_scale_blocks.append(
            ScaleBlock(1, self.d_model, self.num_heads, self.dim_ff, 1, self.dropout_rate, self.in_seg_num,
                       self.factor))
        for i in range(1, self.num_encoder_layers):
            self.encoder_scale_blocks.append(
                ScaleBlock(self.win_size, self.d_model, self.num_heads, self.dim_ff, 1, self.dropout_rate,
                           math.ceil(self.in_seg_num / self.win_size ** i), self.factor))
        self.encoder = Encoder(self.encoder_scale_blocks)

        # decoder
        self.decoder_self_attention_layer = TwoStageAttentionLayer((self.pad_out_len // self.seg_len), self.factor,
                                                                   self.d_model, self.num_heads, self.dim_ff,
                                                                   self.dropout_rate)
        self.decoder_cross_attention_layer = AttentionLayer(FullAttention(self.dropout_rate), self.d_model,
                                                            self.num_heads)
        self.decoder_layer = DecoderLayer(self.decoder_self_attention_layer, self.decoder_cross_attention_layer,
                                          self.seg_len, self.d_model, self.dropout_rate)
        self.decoder_layers = [self.decoder_layer for _ in range(self.num_decoder_layers)]
        self.decoder = Decoder(self.decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: input tensor of shape (batch_size, input_window_size, input_vars)
            :return: output tensor of shape (batch_size, output_window_size, input_vars)
        """
        batch_size, input_window_size, input_vars = x.shape
        x_patches = self.patch_maker(x)  # -> (batch_size, input_vars, patch_num, seg_len)
        x_patches = x_patches.flatten(0, 1)  # -> (batch_size * input_vars, patch_num, seg_len)

        patch_embedding = self.value_embedding(x_patches) + self.encoder_pe(x_patches)  # seg_len -> d_model
        patch_embedding = self.encoder_dropout(patch_embedding)

        # -> (batch_size, input_vars, patch_num, seg_len)
        patch_embedding = patch_embedding.unflatten(0, (batch_size, input_vars))
        patch_embedding += self.encoder_pos_embedding

        encoder_in = self.pre_norm(patch_embedding)  # -> (batch_size, input_vars, patch_num, d_model)
        encoder_out = self.encoder(encoder_in)  # -> (batch_size, input_vars, patch_num, d_model)

        decoder_in = self.decoder_pos_embedding.repeat(batch_size, 1, 1, 1)

        out = self.decoder(decoder_in, encoder_out)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, input_vars)

        return out
