#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class GLU(nn.Module):
    """
        Gated Linear Unit (GLU).
        Language Modeling with Gated Convolutional Networks,
        Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier,
        Proceedings of the 34th International Conference on Machine Learning, PMLR 70:933-941, 2017.
    """

    def __init__(self, input_channel: int, output_channel: int):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x => [..., input_channel] """
        left = self.linear_left(x)  # => [..., output_channel]
        right = torch.sigmoid(self.linear_right(x))  # => [..., output_channel]

        return left * right  # => [..., output_channels]

