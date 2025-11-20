#!/usr/bin/env python
# encoding: utf-8
"""
CRUWrapper - lightweight continuous-time recurrent unit wrapper
Designed to be pyFAST / SMIrDataset friendly:
 - forward(ts, ts_mask, ex_ts2=None)
 - ts: (B, T_in, D)
 - ts_mask: (B, T_in, D)  (1.0 for observed, 0.0 for missing)
 - ex_ts2: optional (B, T_in, time_vars) where time_vars >= 1 and ex_ts2[...,0] contains timestamps
 - returns: (B, L_pred, D)
This implementation is an independent, permissive re-implementation inspired by CRU concepts:
 continuous-time state decay via learnable rates + gated updates.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utility modules / funcs
# -------------------------
def safe_softplus(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softplus (wrap)."""
    return F.softplus(x)


def make_dt_from_ex_ts2(ex_ts2: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Compute per-step delta-times from ex_ts2 timestamps.
    Args:
        ex_ts2: (B, T, time_vars) or (B, T) if user gave only timestamps
    Returns:
        dt: (B, T) where dt[:,0] = 0 and dt[:,t] = timestamps[:,t] - timestamps[:,t-1]
    """
    if ex_ts2 is None:
        raise ValueError("ex_ts2 must not be None in make_dt_from_ex_ts2")

    if ex_ts2.dim() == 2:
        timestamps = ex_ts2
    else:
        timestamps = ex_ts2[..., 0]

    # assume timestamps are monotonic; compute diffs
    dt = torch.zeros_like(timestamps, device=device, dtype=dtype)
    if timestamps.size(1) > 1:
        dt[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
    # clamp negatives to small positive epsilon
    dt = torch.clamp(dt, min=0.0)
    return dt


# -------------------------
# Core continuous RNN cell
# -------------------------
class SimpleCRUCell(nn.Module):
    """
    A lightweight continuous recurrent unit cell inspired by CRU.
    Features:
      - learnable decay rate per hidden dimension (via positive parameter alpha)
      - gated update combining previous hidden state and new input (masked)
      - vectorized over batch and variables
    Expected input shapes:
      obs_t: (B, D)  - observed values at time t (missing values should be zeroed by caller using mask)
      mask_t: (B, D) - binary mask (1 observed, 0 missing)
      h_prev: (B, D, H) or (B, H) depending on broadcast mode
      dt_t: (B,) or (B, D) - delta time to next step
    This implementation will use hidden size H == feature dimension if desired.
    """

    def __init__(self, input_dim: int, hidden_dim: int, use_input_gate: bool = True):
        super().__init__()
        self.input_dim = input_dim  # D (number of observed variables)
        self.hidden_dim = hidden_dim  # H
        # decay parameter: positive via softplus
        self.log_alpha = nn.Parameter(torch.randn(hidden_dim) * 0.1)  # shape (H,)
        # linear maps (vectorized): from obs -> hidden update, from hidden prev -> candidate
        self.W_x = nn.Linear(input_dim, hidden_dim, bias=True)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # gating
        self.use_input_gate = use_input_gate
        if use_input_gate:
            self.gate = nn.Sequential(
                nn.Linear(input_dim + hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        # small bias to candidate
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))

        # initialization helper
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        if self.W_x.bias is not None:
            nn.init.zeros_(self.W_x.bias)
        nn.init.xavier_uniform_(self.W_h.weight)
        nn.init.zeros_(self.candidate_bias)

    def forward(self, obs_t: torch.Tensor, mask_t: torch.Tensor, h_prev: torch.Tensor,
                dt_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_t: (B, D)
            mask_t: (B, D) binary 0/1 float tensor
            h_prev: (B, H)
            dt_t: (B,) or (B, 1) nonnegative
        Returns:
            h_new: (B, H)
        """
        # ensure shapes
        B = obs_t.size(0)
        device = obs_t.device
        dtype = obs_t.dtype

        # Compute decay factor per hidden dim: alpha = softplus(log_alpha), decay = exp(-alpha * dt)
        alpha = safe_softplus(self.log_alpha).to(device=device, dtype=dtype)  # (H,)
        # dt_t: (B, 1)
        dt = dt_t.view(B, 1).to(device=device, dtype=dtype)
        decay = torch.exp(-alpha.view(1, -1) * dt)  # (B, H)

        # Candidate update driven by obs and previous hidden
        # Map obs -> hidden space
        x_mapped = self.W_x(obs_t)  # (B, H)
        h_mapped = self.W_h(h_prev)  # (B, H)
        candidate = torch.tanh(x_mapped + h_mapped + self.candidate_bias)  # (B, H)

        # If input gating enabled, compute gate from [obs, h_prev]
        if self.use_input_gate:
            gate_input = torch.cat([obs_t, h_prev], dim=-1)  # (B, D+H)
            g = self.gate(gate_input)  # (B, H)
        else:
            g = torch.ones_like(candidate)

        # Combine: continuous-time decay interpolation
        # h_cont = decay * h_prev + (1 - decay) * candidate
        h_cont = decay * h_prev + (1.0 - decay) * candidate

        # When observation is missing (mask==0), we preserve previous state for the corresponding variable contributions.
        # Since the cell is shared across variables, we apply a heuristic: if a full timestep has no observed variables,
        # we still do continuous decay; otherwise gating already uses obs_t which was zeroed by caller.
        # Return gated mix: g * h_cont + (1-g) * h_prev
        h_new = g * h_cont + (1.0 - g) * h_prev
        return h_new


class CRU(nn.Module):
    """
    CRUWrapper: high-level wrapper exposing forward(ts, ts_mask, ex_ts2=None)
    - ts: (B, T_in, D)
    - ts_mask: (B, T_in, D)
    - ex_ts2: optional (B, T_in, time_vars) where ex_ts2[...,0] are timestamps
    Returns: (B, L_pred, D)
    Notes:
      - This wrapper uses an internal SimpleCRUCell per variable-vectorized hidden state.
      - Hidden dimension is set to hidden_dim (not necessarily equal to D); decoder maps hidden -> D.
      - Trainer should set model.output_window_size (int) before forward if multi-step output required.
    """

    def __init__(self,
                 input_vars: int,
                 hidden_size: int = 64,
                 use_input_gate: bool = True,
                 decoder_hidden: int = 64,
                 # output_layer: str = "linear"
                 ):
        super().__init__()
        self.input_vars = input_vars  # D
        self.hidden_size = hidden_size  # H
        self.cell = SimpleCRUCell(input_dim=input_vars, hidden_dim=hidden_size, use_input_gate=use_input_gate)
        # decoder: hidden -> (D) output per time step
        # We'll use a small MLP that maps hidden -> D
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, decoder_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(decoder_hidden, input_vars)
        )
        self.output_window_size = 1  # can be overridden by Trainer
        # dynamic attributes (kept for compatibility)
        self.input_window_size = None
        self.batch_size = None
        # output layer type not used heavily here, kept for API parity
        # self.output_layer = output_layer

    def init_hidden(self, B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Initialize hidden state h0: (B, H)"""
        h0 = torch.zeros(B, self.hidden_size, device=device, dtype=dtype)
        return h0

    def forward(self, ts: torch.Tensor, ts_mask: torch.Tensor, ex_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        Args:
            ts: (B, T_in, D)
            ts_mask: (B, T_in, D) float 0/1 mask
            ex_ts: (B, T_in, time_vars) or None
        Returns:
            outputs: (B, L_pred, D)
        """
        assert ts.dim() == 3, "ts must be (B, T, D)"

        ts[~ts_mask] = 0.0

        B, T_in, D = ts.shape
        device = ts.device
        dtype = ts.dtype

        self.batch_size = B
        self.input_window_size = T_in
        L_pred = self.output_window_size if self.output_window_size is not None else 1

        # compute dt for each timestep (B, T_in)
        if ex_ts is not None:
            dt_all = make_dt_from_ex_ts2(ex_ts, device=device, dtype=dtype)  # (B, T_in)
        else:
            # assume uniform spacing of 1.0
            dt_all = torch.ones(B, T_in, device=device, dtype=dtype)

        # initial hidden state
        h = self.init_hidden(B, device, dtype)  # (B, H)

        # We'll process the sequence step-by-step. If performance is an issue, consider vectorizing over T.
        for t in range(T_in):
            obs_t = ts[:, t, :]  # (B, D)
            mask_t = ts_mask[:, t, :]  # (B, D)
            # Zero-out missing observations (caller-specified behavior)
            obs_t_z = obs_t * mask_t
            # For dt, we use the time delta since previous observation
            dt_t = dt_all[:, t]  # (B,)
            # The cell expects obs shape (B, D) and returns hidden (B, H)
            h = self.cell(obs_t_z, mask_t, h, dt_t)

        # Now h is the encoded representation (B, H). We'll produce L_pred steps by autoregressive decoding.
        # For simplicity, we use deterministic decoder repeating the last hidden state; a more advanced design
        # could propagate h with predicted observations + dt for each step.
        # Prepare repeated hidden states: (B, L_pred, H)
        h_rep = h.unsqueeze(1).repeat(1, L_pred, 1)
        # Optionally, augment with time encoding for prediction steps (not provided here)
        # Decode each step
        outputs = self.decoder(h_rep)  # (B, L_pred, D)

        return outputs
