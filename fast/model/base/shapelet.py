#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class ShapeletRepresentation(nn.Module):
    def __init__(self, input_window_size: int, input_size: int, output_window_size: int = 1,
                 shapelet_size: int = 3, dropout: float = 0.):
        super(ShapeletRepresentation, self).__init__()

        self.input_window_size = input_window_size
        self.input_size = input_size
        self.output_window_size = output_window_size
        self.shapelet_size = shapelet_size

        # shapelet candidates
        self.candidates = nn.Parameter(torch.zeros(input_window_size, shapelet_size))
        self.l1 = nn.Linear(self.shapelet_size, self.output_window_size)
        self.dropout = nn.Dropout(p=dropout)

        # nn.init.xavier_uniform_(self.candidates)
        # nn.init.xavier_uniform_(self.candidates, gain=nn.init.calculate_gain('relu'))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        # x => [batch_size, window_size, input_size]

        b = x.permute(0, 2, 1)  # => [batch_size, input_size, window_size]

        c = self.candidates.permute(1, 0)  # => [window_size, shapelet_size]

        ret = (b @ c).softmax(dim=2)  # => [batch_size, input_size, shapelet_size]

        ret = self.l1(ret)  # => [batch_size, input_size, output_window_size]

        ret = ret.permute(0, 2, 1)  # => [batch_size, input_size, output_window_size]

        return ret
