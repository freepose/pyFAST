#!/usr/bin/env python
# encoding: utf-8

"""
    Patch maker for input window time series.
"""

import torch
import torch.nn as nn


class PatchMaker:
    """
        Patch maker for input window time series.
        The patch is a sliding window on the window time series.
        There may have overlapping between two adjacent patches, if ``stride`` < ``seq_len`` - ``patch_len``.
        :param seq_len: window sequence length.
        :param patch_len: length of the patch. Commonly, set ``patch_len`` to ``input_window_size`` // 2.
        :param patch_stride: the spacing between two consecutive patches.
        :param padding: padding for the input tensor. Commonly, set ``padding`` to ``patch_stride``.
    """

    def __init__(self, seq_len: int, patch_len: int = 1, patch_stride: int = 1, padding: int = 0):
        assert patch_len > 0, 'The patch length must be greater than 0.'
        assert patch_stride > 0, 'The patch stride must be greater than 0.'
        assert padding >= 0, 'The padding must be greater than or equal to 0.'

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.padding = padding

        self.padding_layer = nn.ReplicationPad1d((0, self.padding))
        self.patch_num: int = (self.seq_len + self.padding - self.patch_len) // self.patch_stride + 1
        assert self.patch_num > 0, "No patches can be generated."

    def __len__(self):
        return self.patch_num

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
            Make patches based on ``x``.
            :param x: the input tensor, shape is (..., seq_len, n_vars).
        """
        x = x.transpose(-2, -1)  # -> (..., n_vars, seq_len)

        if self.padding > 0:
            if x.dtype == torch.bool:
                last = x[..., -1:].expand(*x.shape[:-1], self.padding)
                x = torch.cat((x, last), dim=-1)  # -> (..., n_vars, seq_len + padding)
            else:
                x = self.padding_layer(x)  # -> (..., n_vars, seq_len + padding)

        ## The ``unfold`` method is not used here because it does not support padding in some devices (e.g., MPS).
        # patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)

        start_indices = torch.arange(0, self.patch_num * self.patch_stride, self.patch_stride, device=x.device)
        window_indices = start_indices.unsqueeze(1) + torch.arange(self.patch_len, device=x.device)
        patches = x[..., window_indices]  # -> (..., n_vars, patch_num, patch_len)

        return patches
