#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

class DynamicTanh(nn.Module):
    """
        Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, Zhuang Liu,
        Transformers without Normalization, 2025
        https://arxiv.org/abs/2503.10622

        Dynamic Tanh layer norm.

        :param dim: int, the dimension of the input tensor.
        :param init_alpha: float, the initial value of alpha.
    """
    def __init__(self, dim: int, init_alpha: float = 1.):
        super(DynamicTanh, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
            :param x: torch.Tensor, input tensor of shape (batch_size, seq_len, dim).
        """

        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta
