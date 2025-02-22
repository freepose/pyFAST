#!/usr/bin/env python
# encoding: utf-8

import math
from typing import List, Literal

import torch
import torch.nn as nn

from .embedding import TokenEmbedding
from ...base import get_activation_cls
from ...base.decomposition import DecomposeSeries
from ....data import PatchMaker, InstanceScale, InstanceStandardScale


class DFTDecomposeSeries(nn.Module):
    """
        Fourier transform series decomposition module
        fix bug at ``top_k_freq, top_list = torch.topk(freq, 5)``.
    """

    def __init__(self, top_k: int):
        super(DFTDecomposeSeries, self).__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
        Bottom-up mixing season pattern
    """

    def __init__(self, input_window_size: int, down_sampling_window: int, down_sampling_layers: int):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(input_window_size // (down_sampling_window ** i),
                                    input_window_size // (down_sampling_window ** (i + 1))),
                    nn.GELU(),
                    torch.nn.Linear(input_window_size // (down_sampling_window ** (i + 1)),
                                    input_window_size // (down_sampling_window ** (i + 1))),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list: list):
        # mixing high->low

        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
        Top-down mixing trend pattern
    """

    def __init__(self, input_window_size: int, down_sampling_window: int, down_sampling_layers: int):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(input_window_size // (down_sampling_window ** (i + 1)),
                                    input_window_size // (down_sampling_window ** i)),
                    nn.GELU(),
                    torch.nn.Linear(input_window_size // (down_sampling_window ** i),
                                    input_window_size // (down_sampling_window ** i)),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list: list):
        # mixing low->high

        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, input_window_size: int, output_window_size: int,
                 d_model: int, dim_ff: int, dropout_rate: float,
                 moving_avg: int,
                 top_k: int,
                 channel_independence: bool,
                 decomp_method: str,
                 down_sampling_window: int,
                 down_sampling_layers: int):
        super(PastDecomposableMixing, self).__init__()
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = DecomposeSeries(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFTDecomposeSeries(top_k)
        else:
            raise ValueError('decompsition is error')

        if not channel_independence:
            self.cross_layer = nn.Sequential(nn.Linear(in_features=d_model, out_features=dim_ff),
                                             nn.GELU(),
                                             nn.Linear(in_features=dim_ff, out_features=d_model),)

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            input_window_size, down_sampling_window, down_sampling_layers)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            input_window_size, down_sampling_window, down_sampling_layers)

        self.out_cross_layer = nn.Sequential(nn.Linear(in_features=d_model, out_features=dim_ff),
                                             nn.GELU(),
                                             nn.Linear(in_features=dim_ff, out_features=d_model),)

    def forward(self, x_list: list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])

        return out_list


class TimeMixer(nn.Module):
    """
        Shiyu Wang ,Haixu Wu ,Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Y. Zhang, Jun Zhou
        TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting
        ICLR 2024. url: https://openreview.net/pdf?id=7oLshfEIC2

        TS-Library code: https://github.com/thuml/Time-Series-Library/blob/main/models/TimeMixer.py (our implementation)

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention number.
        :param num_encoder_layers: number of encoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
        :param moving_avg: window size of moving average.
        :param top_k: top k.
        :param channel_independence: False: channel dependence True: channel independence.
        :param decomposition_method: method of series decomposition.
        :param down_sampling_method: down sampling method.
        :param down_sampling_window: down sampling window size.
        :param down_sampling_layers: down sampling layer number.
        :param use_instance_scale: whether to use instance standard scale (a.k.a., RevIN).
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 d_model: int = 16, num_heads: int = 8, num_encoder_layers: int = 1,
                 dim_ff: int = 64, dropout_rate: float = 0.05,
                 moving_avg: int = 25,
                 top_k: int = 5,
                 channel_independence: bool = True,
                 decomposition_method: Literal['moving_avg', 'dft_decomp'] = 'moving_avg',
                 down_sampling_method: Literal['max', 'avg', 'conv'] = 'avg',
                 down_sampling_window: int = 1,
                 down_sampling_layers: int = 1,
                 use_instance_scale: bool = True):
        super(TimeMixer, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate
        self.moving_avg = moving_avg
        self.top_k = top_k
        self.channel_independence = channel_independence
        self.decomp_method = decomposition_method
        self.down_sampling_method = down_sampling_method
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

        self.inst_scale_layers = torch.nn.ModuleList(
            [InstanceStandardScale(input_vars) if use_instance_scale else InstanceScale()
                 for _ in range(down_sampling_layers + 1)])

        self.decompose_series = DecomposeSeries(moving_avg)

        self.value_embedding = TokenEmbedding(1 if channel_independence else input_vars, d_model)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(input_window_size, output_window_size,
                                                                d_model, dim_ff, dropout_rate,
                                                                moving_avg,
                                                                top_k,
                                                                channel_independence,
                                                                decomposition_method,
                                                                down_sampling_window,
                                                                down_sampling_layers
                                                                ) for _ in range(num_encoder_layers)])

        self.predict_layers = torch.nn.ModuleList([
            torch.nn.Linear(input_window_size // (down_sampling_window ** i),
                            output_window_size)
            for i in range(down_sampling_layers + 1)
        ])

        if channel_independence:
            self.projection_layer = nn.Linear(d_model, 1)
        else:
            self.projection_layer = nn.Linear(d_model, output_vars)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(input_window_size // (down_sampling_window ** i),
                                input_window_size // (down_sampling_window ** i))
                for i in range(down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList([
                torch.nn.Linear(input_window_size // (down_sampling_window ** i),
                                output_window_size)
                for i in range(down_sampling_layers + 1)
            ])

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size, input_size] """
        batch_size, input_window_size, input_size = x.shape

        # multi scale process inputs
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        else:
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.input_vars, out_channels=self.input_vars,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False, dtype=x.dtype, device=x.device)

        x_original = x.transpose(1, 2)  # -> (batch_size, input_size, input_window_size)
        x_sampling_list = [x_original.transpose(1, 2), ]

        for i in range(self.down_sampling_layers):
            x_sampling = down_pool(x_original)
            x_sampling_list.append(x_sampling.transpose(1, 2))
            x_original = x_sampling

        x = x_sampling_list

        # instance scale
        x_list = []
        for i, x in zip(range(len(x)), x):
            # x -> [batch_size, input_window_size, feature_size]
            x = self.inst_scale_layers[i].fit_transform(x)
            if self.channel_independence:
                x = x.transpose(1, 2)  # -> (batch_size, feature_size, input_window_size)
                x = x.flatten(0, 1)  # -> (batch_size * feature_size, input_window_size)
                x = x.unsqueeze(2)  # -> (batch_size * feature_size, input_window_size, 1)
            x_list.append(x)

        # pre-encode
        if self.channel_independence:
            x_list = (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.decompose_series(x)  # preprocess
                out1_list.append(x_1)
                out2_list.append(x_2)
            x_list = (out1_list, out2_list)

        # embedding
        x_embedding_list = []
        for i, x in zip(range(len(x_list[0])), x_list[0]):
            embedding_out = self.value_embedding(x)
            embedding_out = self.embedding_dropout(embedding_out)
            x_embedding_list.append(embedding_out)

        # past decomposable mixing as encoder for past
        encoder_out_list = x_embedding_list
        for i in range(self.num_encoder_layers):
            encoder_out_list = self.pdm_blocks[i](encoder_out_list)

        # future multipredictor mixing as decoder for future
        decoder_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), encoder_out_list):
                dec_out = enc_out.transpose(1, 2)
                dec_out = self.predict_layers[i](dec_out)
                dec_out = dec_out.transpose(1, 2)

                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(batch_size, self.output_vars, self.output_window_size)
                dec_out = dec_out.transpose(1, 2)

                decoder_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), encoder_out_list, x_list[1]):
                dec_out = enc_out.transpose(1, 2)
                dec_out = self.predict_layers[i](dec_out)
                dec_out = dec_out.transpose(1, 2)

                dec_out = self.projection_layer(dec_out)
                out_res = out_res.transpose(1, 2)
                out_res = self.out_res_layers[i](out_res)
                out_res = self.regression_layers[i](out_res)
                out_res = out_res.transpose(1, 2)
                dec_out = dec_out + out_res

                decoder_out_list.append(dec_out)

        out = torch.stack(decoder_out_list, dim=-1).sum(-1)
        out = self.inst_scale_layers[0].inverse_transform(out)

        return out
