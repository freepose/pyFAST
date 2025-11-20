#!/usr/bin/env python
# coding=utf-8

from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from ....model.mts.transformer.embedding import TokenEmbedding, PositionalEncoding
from ...base.activation import get_activation_cls


class ProbMask:
    def __init__(self, B: int, H: int, L: int, index: int, scores: torch.Tensor, device: str or torch.device):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
                    torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index,
                    :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbAttention(nn.Module):
    """
        Probabilistic attention mechanism for Informer.
        :param mask_flag: Whether to apply a masking operation to mask out invalid attention scores.
        :param factor: A scaling factor that controls the ratio between the number of samples in Q and K and the sequence length.
        :param scale: A scaling factor for adjusting the attention scores.
        :param dropout: Dropout rate.
        :param output_attention: Whether to return the computed attention matrix.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: float = None, dropout: float = 0.1,
                 output_attention: bool = False):
        super(ProbAttention, self).__init__()
        self.mask_flag = mask_flag
        self.factor = factor
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = \
            torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c * ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c * ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


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


class ConvLayer(nn.Module):
    """
        Convolution layer for Transformer series.
    """

    def __init__(self, c_in: int):
        """
            :param c_in: number of input variables.
        """
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
        Encoder layer for Transformer series.
    """

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = "relu"):
        """
            :param attention: the type of attention mechanism used.
            :param d_model: model dimension, a.k.a., embedding size.
            :param d_ff: feed forward dimension.
            :param dropout: dropout rate.
            :param activation: activation function.
        """
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
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
    """

    def __init__(self, attn_layers: list, conv_layers: list, norm_layer: nn.Module):
        """
            :param attn_layers: attention layer.
            :param conv_layers: convolution layer.
            :param norm_layer: normalization layer.
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        # x [B, L, D]
        attns = []
        for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
            x, attn = attn_layer(x, attn_mask=attn_mask)
            x = conv_layer(x)
            attns.append(attn)
        x, attn = self.attn_layers[-1](x)
        attns.append(attn)
        x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    """
        Decoder layer for Transformer series.
    """

    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, d_ff: int = None,
                 dropout: float = 0.1, activation: str = "relu"):
        """
            :param self_attention: the type of self attention mechanism used.
            :param cross_attention: the type of cross attention mechanism used.
            :param d_model: model dimension, a.k.a., embedding size.
            :param d_ff: feed forward dimension.
            :param dropout: dropout rate.
            :param activation: activation function.
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    """
        Decoder for Transformer series.
    """

    def __init__(self, layers: list, norm_layer: nn.Module = None, projection: nn.Module = None):
        """
            :param layers: attention layer.
            :param norm_layer: normalization layer.
            :param projection: full connected projection layer.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        x = self.norm(x)
        x = self.projection(x)
        return x


class InformerMaskEx2(nn.Module):
    """
        Zhou H, Zhang S, Peng J, et al.
        Informer: Beyond efficient transformer for long sequence time-series forecasting [C]
        Proceedings of the 35th AAAI conference on artificial intelligence. 2021, 35(12): 11106 -- 11115.
        Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132

        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param ex2_vars: number of exogenous2 (a.k.a., pre-known) variables.
        :param label_window_size: label window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param num_decoder_layers: number of decoder layers.
        :param dim_ff: feed forward dimension.
        :param activation: activation unit function name.
        :param factor: A scaling factor that controls the ratio between the number of samples in Q and K and the sequence length.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_vars: int = 1, output_vars: int = 1, ex2_vars: int = None,
                 label_window_size: int = None,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 1,
                 dim_ff: int = 2048,
                 activation: str = 'gelu',
                 factor: int = 3,
                 dropout_rate: float = 0.05):
        super(InformerMaskEx2, self).__init__()
        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_vars = input_vars
        self.input_window_size = None  # to be set/update in forward()
        self.output_window_size = None  # to be set/update in forward()
        self.output_vars = output_vars
        self.label_window_size = label_window_size
        self.ex2_vars = ex2_vars

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.activation = activation
        self.factor = factor
        self.dropout_rate = dropout_rate

        # encoder_embedding
        self.encoder_value_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.encoder_position_embedding = PositionalEncoding(self.d_model)
        if self.ex2_vars:
            self.encoder_temporal_embedding = nn.Linear(self.ex2_vars, self.d_model, bias=False)
        self.encoder_dropout = nn.Dropout(self.dropout_rate)

        # decoder_embedding
        self.decoder_value_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.decoder_position_embedding = PositionalEncoding(self.d_model)
        if self.ex2_vars:
            self.decoder_temporal_embedding = nn.Linear(self.ex2_vars, self.d_model, bias=False)
        self.decoder_dropout = nn.Dropout(self.dropout_rate)

        # encoder
        self.encoder_self_attention = ProbAttention(False, self.factor, dropout=self.dropout_rate)
        self.encoder_self_attention_layer = AttentionLayer(self.encoder_self_attention, self.d_model, self.num_heads)
        self.encoder_attention_layer = EncoderLayer(self.encoder_self_attention_layer, self.d_model,
                                                    self.dim_ff, self.dropout_rate, activation=self.activation)
        self.encoder_attention_layers = [self.encoder_attention_layer for _ in range(self.num_encoder_layers)]

        self.encoder_conv_layer = ConvLayer(self.d_model)
        self.encoder_conv_layers = [self.encoder_conv_layer for _ in range(self.num_encoder_layers - 1)]

        self.encoder_norm_layer = torch.nn.LayerNorm(self.d_model)
        self.encoder = Encoder(attn_layers=self.encoder_attention_layers,
                               conv_layers=self.encoder_conv_layers,
                               norm_layer=self.encoder_norm_layer)

        # decoder
        self.decoder_self_attention = ProbAttention(True, self.factor, dropout=self.dropout_rate)
        self.decoder_self_attention_layer = AttentionLayer(self.decoder_self_attention, self.d_model, self.num_heads)

        self.decoder_cross_attention = ProbAttention(False, self.factor, dropout=self.dropout_rate)
        self.decoder_cross_attention_layer = AttentionLayer(self.decoder_cross_attention, self.d_model, self.num_heads)

        self.decoder_layer = DecoderLayer(self.decoder_self_attention_layer,
                                          self.decoder_cross_attention_layer,
                                          self.d_model, d_ff=self.dim_ff,
                                          dropout=self.dropout_rate, activation=self.activation)

        self.decoder_layers = [self.decoder_layer for _ in range(self.num_decoder_layers)]
        self.decoder_norm_layer = nn.LayerNorm(self.d_model)
        self.decoder_projection = nn.Linear(self.d_model, self.output_vars)
        self.decoder = Decoder(self.decoder_layers, self.decoder_norm_layer, self.decoder_projection)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None, ex2: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input window time series, shape is (batch, input_window_size, input_vars).
            :param ex2: [Optional] temporal features for the input and output windows,
                        shape is (batch, input_window_size + output_window_size, temporal_vars).
                        The time features are **preknown** information.

                        If the temporal features are not provided, then works as a standard Transformer model.
            :return: output window time series,
                     shape is (batch, output_window_size <= input_window_size, output_vars == input_vars).
        """
        self.label_window_size = self.input_window_size // 2

        xe = x
        if x_mask is not None:
            xe[~x_mask] = 0  # set nan values to zeros.

        if ex2 is not None:
            ex2_inputs = ex2[:, :-self.output_window_size, :]
            ex2_outputs = ex2[:, -self.label_window_size - self.output_window_size:, :]

        x_embedding = self.encoder_value_embedding(xe) + self.encoder_position_embedding(xe) # -> (batch_size, input_window_size, d_model)
        if self.ex2_vars and ex2 is not None:
            x_embedding += self.encoder_temporal_embedding(ex2_inputs)
        x_embedding = self.encoder_dropout(x_embedding)

        # target -> (batch_size, label_window_size + output_window_size, input_vars)
        batch_size, seq_len, input_vars = x.shape
        target_shape = (batch_size, self.label_window_size + self.output_window_size, input_vars)
        target = torch.zeros(*target_shape, dtype=x.dtype, device=x.device)
        target[:, :self.label_window_size, :] = xe[:, -self.label_window_size:, :]

        # target_embedding -> (batch_size, label_window_size + output_window_size, d_model)
        target_embedding = self.decoder_value_embedding(target) + self.decoder_position_embedding(target)
        if self.ex2_vars and ex2 is not None:
            target_embedding += self.decoder_temporal_embedding(ex2_outputs)
        target_embedding = self.decoder_dropout(target_embedding)

        encoder_out, _ = self.encoder(x_embedding)
        out = self.decoder(target_embedding, encoder_out)  # -> (batch_size, output_window_size, output_vars)
        out = out[:, -self.output_window_size:, :]

        return out
