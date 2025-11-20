#!/usr/bin/env python
# encoding: utf-8

import math
import torch

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from typing import List, Optional


class GaussianFourierFeatureTransform(nn.Module):
    """
    https://github.com/ndahlquist/pytorch-fourier-feature-networks
    Given an input of size [..., time, dim], returns a tensor of size [..., n_fourier_feats, time].
    """

    def __init__(self, input_dim: int, n_fourier_feats: int, scales: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales

        n_scale_feats = n_fourier_feats // (2 * len(scales))
        assert n_scale_feats * 2 * len(scales) == n_fourier_feats, \
            f"n_fourier_feats: {n_fourier_feats} must be divisible by 2 * len(scales) = {2 * len(scales)}"
        B_size = (input_dim, n_scale_feats)
        B = torch.cat([torch.randn(B_size) * scale for scale in scales], dim=1)
        self.register_buffer('B', B)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() >= 2, f"Expected 2 or more dimensional input (got {x.dim()}D input)"
        time, dim = x.shape[-2], x.shape[-1]

        assert dim == self.input_dim, \
            f"Expected input to have {self.input_dim} channels (got {dim} channels)"

        # x = torch.einsum('... t n, n d -> ... t d', [x, self.B])
        x = torch.einsum('... t n, n d -> ... t d', x, self.B)

        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class INRLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 dropout: Optional[float] = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: Tensor) -> Tensor:
        out = self._layer(x)
        return self.norm(out)

    def _layer(self, x: Tensor) -> Tensor:
        return self.dropout(torch.sin(self.linear(x)))


class INR(nn.Module):
    def __init__(self, in_feats: int, layers: int, layer_size: int, n_fourier_feats: int, scales: List[int],
                 dropout: Optional[float] = 0.1):
        super().__init__()

        self.features = nn.Linear(in_feats, layer_size) if n_fourier_feats == 0 \
            else GaussianFourierFeatureTransform(in_feats, n_fourier_feats, scales)
        in_size = layer_size if n_fourier_feats == 0 \
            else n_fourier_feats
        layers = [INRLayer(in_size, layer_size, dropout=dropout)] + \
                 [INRLayer(layer_size, layer_size, dropout=dropout) for _ in range(layers - 1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)


class RidgeRegressor(nn.Module):
    def __init__(self, lambda_init: Optional[float] = 0.):
        super().__init__()
        self._lambda = nn.Parameter(torch.as_tensor(lambda_init, dtype=torch.float))

    def forward(self, reprs: Tensor, x: Tensor, reg_coeff: Optional[float] = None):
        if reg_coeff is None:
            reg_coeff = self.reg_coeff()
        w, b = self.get_weights(reprs, x, reg_coeff)
        return w, b

    def get_weights(self, X: Tensor, Y: Tensor, reg_coeff: float):
        batch_size, n_samples, n_dim = X.shape
        ones = torch.ones(batch_size, n_samples, 1, device=X.device)
        X = torch.concat([X, ones], dim=-1)

        if n_samples >= n_dim:
            # standard
            A = torch.bmm(X.mT, X)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            B = torch.bmm(X.mT, Y)

            weights = torch.linalg.solve(A, B)

        else:
            # Woodbury
            A = torch.bmm(X, X.mT)
            A.diagonal(dim1=-2, dim2=-1).add_(reg_coeff)
            weights = torch.bmm(X.mT, torch.linalg.solve(A, Y))

        return weights[:, :-1], weights[:, -1:]

    def reg_coeff(self) -> Tensor:
        return F.softplus(self._lambda)


# class DeepTIMe(nn.Module):
#     def __init__(self, datetime_feats: int, layer_size: int, inr_layers: int, n_fourier_feats: int, scales: float):
#         super().__init__()
#
#         self.inr = INR(in_feats=datetime_feats + 1, layers=inr_layers, layer_size=layer_size,
#                        n_fourier_feats=n_fourier_feats, scales=scales)
#         self.adaptive_weights = RidgeRegressor()
#
#         self.datetime_feats = datetime_feats
#         self.inr_layers = inr_layers
#         self.layer_size = layer_size
#         self.n_fourier_feats = n_fourier_feats
#         self.scales = scales
#
#     def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
#         tgt_horizon_len = y_time.shape[1]
#         batch_size, lookback_len, _ = x.shape
#         coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)
#         if y_time.shape[-1] != 0:
#             time = torch.cat([x_time, y_time], dim=1)
#             # coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
#             coords = coords.repeat(time.shape[0], 1, 1)
#
#             coords = torch.cat([coords, time], dim=-1)
#
#             time_reprs = self.inr(coords)
#         else:
#             # time_reprs = repeat(self.inr(coords), '1 t d -> b t d', b=batch_size)
#             time_reprs = self.inr(coords).repeat(batch_size, 1, 1)
#
#         lookback_reprs = time_reprs[:, :-tgt_horizon_len]
#         horizon_reprs = time_reprs[:, -tgt_horizon_len:]
#         w, b = self.adaptive_weights(lookback_reprs, x)
#         preds = self.forecast(horizon_reprs, w, b)
#
#         return preds
#
#     def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
#         return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b
#
#     def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
#         coords = torch.linspace(0, 1, lookback_len + horizon_len)
#         coords = coords.unsqueeze(1).unsqueeze(0)  # rearrange(coords, 't -> 1 t 1')
#
#         return coords


class DeepTIMe(nn.Module):
    """
        DeepTIMe model (pyFAST compatible, Ex2 variant with pre-known features)

        Woo, Gerald and Liu, Chenghao and Sahoo, Doyen and Kumar, Akshat and Hoi, Steven,
        Learning Deep Time-index Models for Time Series Forecasting (ICML 2023),
        url: https://proceedings.mlr.press/v202/woo23b/woo23b.pdf

        Author provided code: https://github.com/salesforce/DeepTime
    """

    def __init__(self, input_vars: int, output_window_size: int = 96, output_vars: int = 1,
                 ex2_vars: int = None,
                 layer_size: int = 256, inr_layers: int = 3,
                 n_fourier_feats: int = 128, scales: List[int] = [1, 2, 4, 8],
                 dropout_rate: float = 0.1):
        super().__init__()

        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.ex2_vars = ex2_vars

        self.layer_size = layer_size
        self.inr_layers = inr_layers
        self.n_fourier_feats = n_fourier_feats
        self.scales = scales
        self.dropout_rate = dropout_rate

        # Time embedding (like ex2)
        if ex2_vars:
            self.time_embedding = nn.Linear(ex2_vars, 1, bias=False)

        self.inr = INR(in_feats=1 + (1 if ex2_vars else 0),
                       layers=inr_layers,
                       layer_size=layer_size,
                       n_fourier_feats=n_fourier_feats,
                       scales=scales,
                       dropout=dropout_rate)

        self.regressor = RidgeRegressor()

    def forward(self, x: torch.Tensor, ex2: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param x: (B, input_window_size, input_vars)
        :param ex2: (B, input_window_size + output_window_size, ex2_vars)
        :return: (B, output_window_size, output_vars)
        """
        B, L_in, D_in = x.shape
        L_out = self.output_window_size
        coords = torch.linspace(0, 1, L_in + L_out, device=x.device).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)

        # add ex2 features if available
        if self.ex2_vars and ex2 is not None:
            t_feats = self.time_embedding(ex2)
            coords = torch.cat([coords, t_feats], dim=-1)

        time_reprs = self.inr(coords)
        lookback_reprs = time_reprs[:, :L_in]
        horizon_reprs = time_reprs[:, L_in:]

        W, b = self.regressor(lookback_reprs, x)
        preds = torch.einsum('...do,...td->...to', W, horizon_reprs) + b

        return preds
