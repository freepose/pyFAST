#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from typing import List, Tuple
from torch.distributions.normal import Normal

from ....data import InstanceStandardScale
from ...base.decomposition import DecomposeSeriesMultiKernels


class FourierLayer(nn.Module):
    """
        Simple Fourier-based seasonal-trend decomposition.
        Given x of shape (B, T, C), keeps top-k frequency components to form seasonality,
        and returns (seasonality, trend) where trend = x - seasonality.
    """

    def __init__(self, pred_len: int = 0, k: int = 3):
        super(FourierLayer, self).__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        # zero DC to avoid dominating selection
        freq[:, :1, :] = 0
        # threshold by top-k frequencies per (B, C)
        top_k_freq, _ = torch.topk(freq, k=min(self.k, freq.size(1)), dim=1)
        thresh = top_k_freq.min(dim=1, keepdim=True).values  # (B, 1, C)
        mask = freq >= thresh  # (B, F, C)
        xf_filtered = torch.where(mask, xf, torch.zeros_like(xf))
        season = torch.fft.irfft(xf_filtered, n=x.size(1), dim=1)
        trend = x - season
        return season, trend


class Transformer_Layer(nn.Module):
    """
    A lightweight temporal Transformer encoder operating on non-overlapping patches along time.
    It reshapes (B, L, N, D) -> patches of size `patch_size`, applies Transformer encoder per patch,
    and reconstructs to (B, L, N, D). Tail part that doesn't fit an integer number of patches is passed through.
    """

    def __init__(self, d_model: int = 32, d_ff: int = 64, num_heads: int = 4,
                 patch_nums: int = 1, patch_size: int = 8, factorized: bool = True,
                 layer_number: int = 1, batch_norm: bool = False, dynamic: bool = False, num_nodes: int = 1):
        super(Transformer_Layer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = max(1, min(num_heads, d_model))
        self.patch_size = patch_size
        self.layer_number = layer_number
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=self.num_heads,
                                                   dim_feedforward=d_ff, batch_first=True, norm_first=False)
        encoder_layer.use_nested_tensor = False
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.use_bn = bool(batch_norm)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(num_nodes)
        else:
            self.bn = None

    def forward(self, x: torch.Tensor):
        # x: (B, L, N, D)
        B, L, N, D = x.shape
        P = max(1, L // self.patch_size)
        T_eff = P * self.patch_size
        head = x[:, :T_eff, :, :]  # (B, T_eff, N, D)
        tail = x[:, T_eff:, :, :] if T_eff < L else None
        # reshape to (B*N*P, patch_size, D)
        h = head.reshape(B, P, self.patch_size, N, D).permute(0, 3, 1, 2, 4).contiguous()  # (B, N, P, K, D)
        h = h.view(B * N * P, self.patch_size, D)
        h = self.encoder(h)  # (B*N*P, K, D)
        h = h.view(B, N, P, self.patch_size, D).permute(0, 2, 3, 1, 4).contiguous()  # (B, P, K, N, D)
        head_out = h.view(B, T_eff, N, D)
        if tail is not None:
            out = torch.cat([head_out, tail], dim=1)
        else:
            out = head_out
        if self.bn is not None:
            # BN over node dimension with (B*L, N, D) -> BN over N
            out_ = out.view(B * L, N, D)
            out_ = self.bn(out_)
            out = out_.view(B, L, N, D)
        return out, None


class SparseDispatcher:
    """
    Dispatcher for sparse mixture-of-experts routing.
    It splits the batch according to non-zero gates, and can combine expert outputs back.
    """

    def __init__(self, num_experts: int, gates: torch.Tensor):
        # gates: (B, E)
        self.num_experts = num_experts
        self.gates = gates
        self.batch_size = gates.size(0)
        self.assignments: List[torch.Tensor] = []
        self.batch_indices: List[torch.Tensor] = []
        for i in range(num_experts):
            mask = gates[:, i] > 0
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            self.batch_indices.append(indices)
            self.assignments.append(gates[indices, i])  # (b_i,)

    def dispatch(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, ...)
        expert_inputs = []
        for indices in self.batch_indices:
            if indices.numel() == 0:
                expert_inputs.append(x[:0])  # empty tensor with same dims
            else:
                expert_inputs.append(x.index_select(dim=0, index=indices))
        return expert_inputs

    def combine(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        # each expert_outputs[i]: (b_i, ...)
        # we will sum weighted outputs for samples that went to multiple experts (when k>1)
        if len(expert_outputs) == 0:
            raise ValueError("No expert outputs to combine.")
        out_shape = (self.batch_size, *expert_outputs[0].shape[1:])
        output = torch.zeros(out_shape, device=expert_outputs[0].device, dtype=expert_outputs[0].dtype)
        for i, indices in enumerate(self.batch_indices):
            if indices.numel() == 0:
                continue
            weights = self.assignments[i].to(expert_outputs[i].dtype)
            # reshape weights for broadcasting across remaining dims
            w_shape = [weights.size(0)] + [1] * (expert_outputs[i].ndim - 1)
            weighted = expert_outputs[i] * weights.view(*w_shape)
            output.index_add_(0, indices, weighted)
        return output


class AMS(nn.Module):
    def __init__(self, input_size, output_size, num_experts: int, num_nodes=1, d_model=32, d_ff=64, dynamic=False,
                 patch_size: List[int] | None = None, noisy_gating=True, k=4, layer_number=1,
                 residual_connection: bool = True, batch_norm=False):
        super(AMS, self).__init__()
        # ensure patch_size is a list
        if patch_size is None:
            patch_size = [8, 6, 4, 2]
        if isinstance(patch_size, int):
            patch_size = [patch_size]
        self.patch_size_list = patch_size
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = min(k, self.num_experts)

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = DecomposeSeriesMultiKernels(3, 7, 11)

        self.experts = nn.ModuleList()
        for patch in self.patch_size_list:
            patch_nums = max(1, int(input_size // patch))
            self.experts.append(Transformer_Layer(d_model=d_model, d_ff=d_ff,
                                                  dynamic=dynamic, num_nodes=num_nodes, patch_nums=patch_nums,
                                                  patch_size=patch, factorized=True, layer_number=layer_number,
                                                  batch_norm=batch_norm))

        # gating parameters
        self.w_noise = nn.Linear(input_size, self.num_experts)
        self.w_gate = nn.Linear(input_size, self.num_experts)

        self.residual_connection = residual_connection

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        # x: (B, L, N, D)
        x0 = x[:, :, :, 0]  # (B, L, N)
        seasonality, _ = self.seasonality_model(x0)
        trend, _ = self.trend_model(x0)
        return x0 + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        x = self.start_linear(x).squeeze(-1)

        clean_logits = self.w_gate(x)
        # compute noise stddev regardless for shape safety (may be unused)
        raw_noise_stddev = self.w_noise(x)
        noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon

        if self.noisy_gating and train:
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            noisy_logits = clean_logits  # placeholder to avoid unbound warnings
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        new_x = self.seasonality_and_trend_decompose(x)

        # multi-scale router
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        # calculate balance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + x
        return output, balance_loss


class Pathformer(nn.Module):
    """
        Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting,
        ICLR 2024
        pdf: https://arxiv.org/abs/2402.05956
        official code: https://github.com/decisionintelligence/pathformer

        Implementation Author: Zhuorui Wu.

        The MPS device (Apple Silicon) is not supported.
    """
    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 layer_nums: int = 3, k: int = 3,
                 num_experts_list: List[int] = None,
                 patch_size_list: List[int] = None,
                 d_model: int = 4,
                 d_ff: int = 64, residual_connection: bool = True, use_instance_scale: bool = True,
                 batch_norm: bool = False):
        super().__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.layer_nums = layer_nums
        self.k = k
        # defaults
        if patch_size_list is None:
            self.patch_size_list = [16, 12, 8, 6, 4, 2]
        if num_experts_list is None:
            self.num_experts_list = [4] * layer_nums
        self.patch_size_list = patch_size_list
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_connection = residual_connection
        self.use_instance_scale = use_instance_scale
        self.batch_norm = bool(batch_norm)
        self.balance_loss = None
        if self.use_instance_scale:
            self.inst_scale = InstanceStandardScale(num_features=input_vars)
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.input_window_size, self.input_window_size, num_experts=self.num_experts_list[num], k=self.k,
                    num_nodes=self.input_vars, patch_size=self.patch_size_list, noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(nn.Linear(self.input_window_size * self.d_model, self.output_window_size))

    def forward(self, x: torch.Tensor):
        """
            :param x: (B, L_in, N)
            :return: (B, L_out, N)
        """
        if self.use_instance_scale:
            x = self.inst_scale.fit_transform(x)

        out = self.start_fc(x.unsqueeze(-1))

        batch_size = x.shape[0]

        self.balance_loss = 0
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            self.balance_loss += aux_loss

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.input_vars, -1)
        out = self.projections(out).transpose(2, 1)

        if self.use_instance_scale:
            out = self.inst_scale.inverse_transform(out)

        return out

    def additional_loss(self, prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.balance_loss
