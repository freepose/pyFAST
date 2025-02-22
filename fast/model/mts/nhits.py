#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from functools import partial
from typing import List, Tuple


class NHiTS(nn.Module):
    """
        Cristian Challu, Kin G. Olivares, Boris N. Oreshkin, Federico Garza, Max Mergenthaler-Canseco, Artur Dubrawski
        N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting, AAAI 2023
        Link: https://arxiv.org/abs/2201.12886
        Official code: https://github.com/cchallu/n-hits

    """

    def __init__(self, input_window_size: int, output_window_size: int,
                 n_blocks: list, n_layers: list, hidden_size: list, pooling_sizes: list,
                 downsample_frequencies: list, pooling_mode: str = 'max', interpolation_mode: str = 'linear',
                 dropout: float = 0.0, activation: str = 'ReLU', initialization: str = 'lecun_normal',
                 batch_normalization: bool = False, shared_weights: bool = False, naive_level: bool = True):
        super().__init__()

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size

        self.context_length = self.input_window_size
        self.prediction_length = self.output_window_size

        self.nhits_output_size = 1
        self.naive_level = naive_level

        blocks = self.create_stack(
            n_blocks=n_blocks,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            output_size=self.nhits_output_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            pooling_sizes=pooling_sizes,
            downsample_frequencies=downsample_frequencies,
            pooling_mode=pooling_mode,
            interpolation_mode=interpolation_mode,
            batch_normalization=batch_normalization,
            dropout=dropout,
            activation=activation,
            shared_weights=shared_weights,
            initialization=initialization,
        )
        self.blocks = torch.nn.ModuleList(blocks)

    def create_stack(self, n_blocks, context_length, prediction_length, output_size,
                     n_layers, hidden_size, pooling_sizes, downsample_frequencies,
                     pooling_mode, interpolation_mode, batch_normalization, dropout,
                     activation, shared_weights, initialization):
        block_list = []
        for i in range(len(n_blocks)):
            for block_id in range(n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list) == 0) and batch_normalization:
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    n_theta = max(prediction_length // downsample_frequencies[i], 1)
                    basis = IdentityBasis(
                        backcast_size=context_length,
                        forecast_size=prediction_length,
                        interpolation_mode=interpolation_mode,
                    )

                    nbeats_block = NHiTSBlock(
                        context_length=context_length,
                        prediction_length=prediction_length,
                        output_size=output_size,
                        n_theta=n_theta,
                        hidden_size=hidden_size[i],
                        pooling_sizes=pooling_sizes[i],
                        pooling_mode=pooling_mode,
                        basis=basis,
                        n_layers=n_layers[i],
                        batch_normalization=batch_normalization_block,
                        dropout=dropout,
                        activation=activation,
                    )

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x -> [batch_size, input_window_size, input_size]"""

        batch_size, input_window_size, input_size = x.shape
        x = x.unsqueeze(-1)  # -> [batch_size, input_window_size, input_size, 1]
        x = x[..., [0]].transpose(1, 2)  # -> [batch_size, input_size, input_window_size, 1]
        x = x.reshape(batch_size * input_size, input_window_size, 1)    # -> [batch_size * input_size, input_window_size, 1]

        residuals = x   # -> [batch_size * input_size, input_window_size, 1]
        level = x[:, -1:].repeat(1, self.prediction_length, 1)  # Level with Naive1 -> [batch_size * input_size, output_window_size, 1]
        forecast_level = level.repeat_interleave(torch.tensor(self.nhits_output_size, device=level.device), dim=2)    # -> [batch_size * input_size, output_window_size, nhits_output_size(1)]

        # level with last available observation
        if self.naive_level:
            block_forecasts = [forecast_level]  # -> list([batch_size * input_size, output_window_size, nhits_output_size(1)])
            block_backcasts = [x[:, -1:].repeat(1, self.context_length, 1)] # -> list([batch_size * input_size, input_window_size, 1])
            forecast = block_forecasts[0]   # -> [batch_size * input_size, output_window_size, nhits_output_size(1)]
        else:
            block_forecasts = []
            block_backcasts = []
            forecast = torch.zeros_like(forecast_level, device=forecast_level.device)

        # forecast by block
        for block in self.blocks:
            # block_backcast -> [batch_size * input_size, input_window_size, 1]
            # block_forecast -> [batch_size * input_size, output_window_size, nhits_output_size(1)]
            block_backcast, block_forecast = block(encoder_y=residuals)
            residuals = (residuals - block_backcast)    # -> [batch_size * input_size, input_window_size, 1]
            forecast = forecast + block_forecast    # -> [batch_size * input_size, output_window_size, nhits_output_size(1)]
            block_forecasts.append(block_forecast)
            block_backcasts.append(block_backcast)

        # (n_batch, n_t, n_outputs, n_blocks)
        block_forecasts = torch.stack(block_forecasts, dim=-1)  # -> [batch_size * input_size, output_window_size, nhits_output_size(1), n_blocks + 1]
        block_backcasts = torch.stack(block_backcasts, dim=-1)  # -> [batch_size * input_size, input_window_size, 1, n_blocks + 1]
        backcast = residuals    # -> [batch_size * input_size, input_window_size, 1, n_blocks + 1]
        # return forecast, backcast, block_forecasts, block_backcasts
        forecast = forecast.reshape(batch_size, input_size, self.prediction_length, self.nhits_output_size).transpose(1, 2)   # -> [batch_size, output_window_size, input_size, nhits_output_size(1)]

        forecast = forecast.squeeze(-1) # -> [batch_size, output_window_size, output_size]
        return forecast


def init_weights(module, initialization):
    if type(module) == torch.nn.Linear:
        if initialization == "orthogonal":
            torch.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            torch.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            torch.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass  # torch.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1 < 0, f"Initialization {initialization} not found"


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ["linear", "nearest"]) or ("cubic" in interpolation_mode)
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(
            self,
            backcast_theta: torch.Tensor,
            forecast_theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        backcast = backcast_theta
        knots = forecast_theta

        if self.interpolation_mode == "nearest":
            knots = knots[:, None, :]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode)
            forecast = forecast[:, 0, :]
        elif self.interpolation_mode == "linear":
            knots = knots[:, None, :]
            forecast = F.interpolate(
                knots, size=self.forecast_size, mode=self.interpolation_mode
            )  # , align_corners=True)
            forecast = forecast[:, 0, :]
        elif "cubic" in self.interpolation_mode:
            batch_size = int(self.interpolation_mode.split("-")[-1])
            knots = knots[:, None, None, :]
            forecast = torch.zeros((len(knots), self.forecast_size)).to(knots.device)
            n_batches = int(np.ceil(len(knots) / batch_size))
            for i in range(n_batches):
                forecast_i = F.interpolate(
                    knots[i * batch_size: (i + 1) * batch_size], size=self.forecast_size, mode="bicubic"
                )  # , align_corners=True)
                forecast[i * batch_size: (i + 1) * batch_size] += forecast_i[:, 0, 0, :]

        return backcast, forecast


class NHiTSBlock(nn.Module):
    def __init__(self, context_length: int, prediction_length: int,
                 output_size: int, n_theta: int, hidden_size: List[int],
                 pooling_sizes: int, pooling_mode: str, basis: nn.Module,
                 n_layers: int, batch_normalization: bool, dropout: float,
                 activation: str):
        super().__init__()

        assert pooling_mode in ["max", "average"]

        self.context_length_pooled = int(np.ceil(context_length / pooling_sizes))

        self.context_length = context_length
        self.output_size = output_size
        self.n_theta = n_theta
        self.prediction_length = prediction_length
        self.pooling_sizes = pooling_sizes
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.activation_list = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]

        self.hidden_size = [self.context_length_pooled] + hidden_size

        assert activation in self.activation_list, f"{activation} is not in {self.activation_list}"
        activ = getattr(nn, activation)()

        if pooling_mode == "max":
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)
        elif pooling_mode == "average":
            self.pooling_layer = nn.AvgPool1d(kernel_size=self.pooling_sizes, stride=self.pooling_sizes, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=self.hidden_size[i], out_features=self.hidden_size[i + 1]))
            hidden_layers.append(activ)

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=self.hidden_size[i + 1]))

            if self.dropout > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout))

        output_layer = [
            nn.Linear(
                in_features=self.hidden_size[-1],
                out_features=context_length + n_theta * output_size,
            )
        ]
        layers = hidden_layers + output_layer

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, encoder_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(encoder_y)

        encoder_y = encoder_y.transpose(1, 2)
        # Pooling layer to downsample input
        encoder_y = self.pooling_layer(encoder_y)
        encoder_y = encoder_y.transpose(1, 2).reshape(batch_size, -1)

        # Compute local projection weights and projection
        theta = self.layers(encoder_y)
        backcast_theta = theta[:, : self.context_length].reshape(-1, self.context_length)
        forecast_theta = theta[:, self.context_length:].reshape(-1, self.n_theta)
        backcast, forecast = self.basis(backcast_theta, forecast_theta)
        backcast = backcast.reshape(-1, 1, self.context_length).transpose(1, 2)
        forecast = forecast.reshape(-1, self.output_size, self.prediction_length).transpose(1, 2)

        return backcast, forecast
