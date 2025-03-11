#!/usr/bin/env python
# coding=utf-8

import math
import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from ...base.activation import get_activation_cls
from ...base.decomposition import DecomposeSeries


class AutoformerLayerNorm(nn.Module):
    """
        Special designed LayerNorm for the seasonal part.
    """

    def __init__(self, channels: int):
        super(AutoformerLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor):
        x_hat = self.norm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class AutoCorrelation(nn.Module):
    """
        AutoCorrelation Mechanism with the following two phases:
        (1) period-based dependencies discovery
        (2) time delay aggregation
        This block can replace the self-attention family mechanism seamlessly.

        :param mask_flag: whether to apply a masking operation to mask out invalid attention scores.
        :param factor: a scaling factor that controls the ratio between the number of samples in Q and K and the sequence length.
        :param scale: a scaling factor for adjusting the attention scores.
        :param dropout_rate: dropout rate.
        :param output_attention: whether to return the computed attention matrix.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 1, scale: float = None, dropout_rate=0.1,
                 output_attention: bool = False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(dropout_rate)
        self.output_attention = output_attention

    def time_delay_agg_training(self, values: torch.Tensor, corr: torch.Tensor):
        """
            SpeedUp version of Autocorrelation (a batch-normalization style design)
            This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values: torch.Tensor, corr: torch.Tensor):
        """
            SpeedUp version of Autocorrelation (a batch-normalization style design)
            This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(
            values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values: torch.Tensor, corr: torch.Tensor):
        """
            Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(
            values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attn_mask: torch.Tensor):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return V.contiguous(), corr.permute(0, 3, 1, 2)
        else:
            return V.contiguous(), None


class AutoCorrelationLayer(nn.Module):
    """
        Auto correlation wrapper layer, which implements multi-head.
    """

    def __init__(self, correlation: nn.Module, d_model: int, n_heads: int,
                 d_keys: int = None, d_values: int = None):
        super(AutoCorrelationLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_correlation = correlation
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

        out, attn = self.inner_correlation(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """
        Autoformer encoder layer with the progressive decomposition architecture

        :param attention: the type of attention mechanism used.
        :param d_model: model dimension, a.k.a., embedding size.
        :param d_ff: feed forward dimension.
        :param moving_avg: the kernel size of series decomposition function。
        :param dropout: dropout rate.
        :param activation: activation function.
    """

    def __init__(self, attention: nn.Module, d_model: int, d_ff: int, moving_avg: int = 25,
                 dropout: float = 0.1, activation: str = "relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = DecomposeSeries(moving_avg)
        self.decomp2 = DecomposeSeries(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)

        return res, attn


class Encoder(nn.Module):
    """
        Autoformer encoder

        :param attn_layers: attention layer.
        :param norm_layer: normalization layer.
    """

    def __init__(self, attn_layers: list, norm_layer: nn.Module):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        x = self.norm(x)
        return x, attns


class DecoderLayer(nn.Module):
    """
        Autoformer decoder layer with the progressive decomposition architecture

        :param self_attention: the type of self attention mechanism used.
        :param cross_attention: the type of cross attention mechanism used.
        :param d_model: model dimension, a.k.a., embedding size.
        :param c_out: the number of output channels of the convolution operation.
        :param d_ff: feed forward dimension.
        :param moving_avg: the kernel size of series decomposition function。
        :param dropout: dropout rate.
        :param activation: activation function.
    """

    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, c_out: int, d_ff: int,
                 moving_avg: int = 25, dropout: float = 0.1, activation: str = "relu"):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = DecomposeSeries(moving_avg)
        self.decomp2 = DecomposeSeries(moving_avg)
        self.decomp3 = DecomposeSeries(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = get_activation_cls(activation)()

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: torch.Tensor = None,
                cross_mask: torch.Tensor = None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
        Autoformer decoder

        :param layers: attention layer.
        :param norm_layer: normalization layer.
        :param projection: full connected projection layer.
    """

    def __init__(self, layers: list, norm_layer: nn.Module, projection: nn.Module):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor,
                x_mask: torch.Tensor = None, cross_mask: torch.Tensor = None, trend: torch.Tensor = None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend
        x = self.norm(x)
        x = self.projection(x)
        return x, trend


class Autoformer(nn.Module):
    """
        Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long.
        Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting.
        Proceedings of the 35th Neural Information Processing Systems 34 (2021): 22419-22430.
        url: https://openreview.net/pdf?id=I55UqU-M11y

        Official Code: https://github.com/thuml/Autoformer
        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/Autoformer.py (our implementation)

        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param label_window_size: label window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param num_decoder_layers: number of decoder layers.
        :param dim_ff: feed forward dimension.
        :param activation: activation unit function name.
        :param moving_avg: the kernel size of series decomposition function。
        :param factor: A scaling factor that controls the ratio between the number of samples in Q and K and the sequence length.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_vars: int = 1, output_window_size: int = 1, output_vars: int = 1,
                 label_window_size: int = 0,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 1,
                 dim_ff: int = 2048,
                 activation: str = 'gelu',
                 moving_avg: int = 25,
                 factor: int = 3,
                 dropout_rate: float = 0.05):
        super(Autoformer, self).__init__()
        assert output_vars == input_vars and 0 <= label_window_size, 'Invalid window parameters.'
        assert dim_ff % num_heads == 0, 'dim_ff should be divided by num_heads.'

        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.label_window_size = label_window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.factor = factor
        self.moving_avg = moving_avg
        self.decompose_series = DecomposeSeries(self.moving_avg)

        # encoder_embedding
        self.encoder_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.encoder_dropout = nn.Dropout(self.dropout_rate)

        # decoder_embedding
        self.decoder_embedding = TokenEmbedding(self.input_vars, self.d_model)
        self.decoder_dropout = nn.Dropout(self.dropout_rate)

        # encoder
        self.encoder_self_autocorrelation = AutoCorrelation(factor=self.factor, scale=None,
                                                            dropout_rate=self.dropout_rate, mask_flag=False)
        self.encoder_self_autocorrelation_layer = AutoCorrelationLayer(correlation=self.encoder_self_autocorrelation,
                                                                       d_model=self.d_model, n_heads=self.num_heads)
        self.encoder_layer = EncoderLayer(attention=self.encoder_self_autocorrelation_layer,
                                          moving_avg=self.moving_avg,
                                          d_model=self.d_model, d_ff=self.dim_ff,
                                          dropout=self.dropout_rate, activation=self.activation)
        self.encoder_layers = [self.encoder_layer for _ in range(self.num_encoder_layers)]
        self.encoder_norm_layer = AutoformerLayerNorm(self.d_model)
        self.encoder = Encoder(attn_layers=self.encoder_layers, norm_layer=self.encoder_norm_layer)

        # decoder
        self.decoder_self_autocorrelation = AutoCorrelation(factor=self.factor, scale=None,
                                                            dropout_rate=self.dropout_rate, mask_flag=True)
        self.decoder_self_autocorrelation_layer = AutoCorrelationLayer(correlation=self.decoder_self_autocorrelation,
                                                                       d_model=self.d_model, n_heads=self.num_heads)
        self.decoder_cross_autocorrelation = AutoCorrelation(factor=self.factor, scale=None,
                                                             dropout_rate=self.dropout_rate, mask_flag=False)
        self.decoder_cross_autocorrelation_layer = AutoCorrelationLayer(correlation=self.decoder_cross_autocorrelation,
                                                                        d_model=self.d_model, n_heads=self.num_heads)
        self.decoder_layer = DecoderLayer(self_attention=self.decoder_self_autocorrelation_layer,
                                          cross_attention=self.decoder_cross_autocorrelation_layer,
                                          moving_avg=self.moving_avg,
                                          d_model=self.d_model, d_ff=self.dim_ff, c_out=self.input_vars,
                                          dropout=self.dropout_rate, activation=self.activation)
        self.decoder_layers = [self.decoder_layer for _ in range(self.num_decoder_layers)]
        self.decoder_norm_layer = AutoformerLayerNorm(self.d_model)
        self.decoder_projection = nn.Linear(self.d_model, self.output_vars, bias=True)
        self.decoder = Decoder(layers=self.decoder_layers, norm_layer=self.decoder_norm_layer,
                               projection=self.decoder_projection)

    def forward(self, x: torch.Tensor):
        """ x -> (batch_size, input_window_size, input_vars) """

        # encoder_embedding
        x_embedding = self.encoder_embedding(x)  # -> [batch_size, input_window_size, d_model]
        x_embedding = self.encoder_dropout(x_embedding)

        # seasonal and trend -> [batch_size, label_window_size + output_window_size, input_vars]
        batch_size, seq_len, input_vars = x.shape
        seasonal_init, trend_init = self.decompose_series(x)
        mean = torch.mean(x, dim=1).unsqueeze(1).repeat(1, self.output_window_size, 1)
        zeros = torch.zeros([batch_size, self.output_window_size, self.input_vars], device=x.device)
        trend_init = torch.cat([trend_init[:, seq_len - self.label_window_size:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, seq_len - self.label_window_size:, :], zeros], dim=1)

        # target embedding -> [batch_size, label_window_size + output_window_size, d_model]
        target_embedding = self.decoder_embedding(seasonal_init)
        target_embedding = self.decoder_dropout(target_embedding)

        encoder_out, attns = self.encoder(x_embedding)
        seasonal_part, trend_part = self.decoder(target_embedding, encoder_out, trend=trend_init)
        out = trend_part + seasonal_part
        out = out[:, -self.output_window_size:, :]

        return out
