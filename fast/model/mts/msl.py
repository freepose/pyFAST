#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from typing import Literal

class MSL(nn.Module):
    """
        MSL. Multivariate Shapelet Learning.

        Wang, Zhijin, Cai, Bing.
        COVID-19 cases prediction in multiple areas via shapelet learning.
        Applied Intelligence 52, 595–606 (2022).
        url: https://doi.org/10.1007/s10489-021-02391-6

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param method: method in [``inner_product`` | ``soft_min_distance``]
        :param shapelet_size: the shaplet size
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 method: Literal['inner_product', 'soft_min_distance'] = 'inner_product', shapelet_size: int = 3):
        super(MSL, self).__init__()
        self.input_window_size = input_window_size
        self.input_size = input_vars
        self.output_window_size = output_window_size

        self.method = method  # [inner_product, soft_min_distance]
        self.shapelet_size = shapelet_size

        gain = 0.01  # 1. / math.sqrt(self.window_size * self.shapelet_size)
        self.centroids = nn.Parameter(torch.rand(self.shapelet_size, self.input_window_size) * gain)
        self.l1 = nn.Linear(self.shapelet_size, self.output_window_size)  # mapping

        self.softmin = nn.Softmin(dim=1)  # softmin on shapelets, which shapelet is more important?

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size, input_size] """

        if self.method == 'inner_product':
            b = x.permute(0, 2, 1)  # -> [batch_size, input_size, input_window_size]
            c = self.centroids.permute(1, 0)  # -> [input_window_size, centroid_size]
            ret = (b @ c).softmax(dim=2)  # -> [batch_size, input_size, centroid_size]
            ret = self.l1(ret)  # -> [batch_size, input_size, centroid_size]
            ret = ret.permute(0, 2, 1)  # -> [batch_size, input_size, output_window_size]
            # ret = torch.relu(ret)

            return ret
        elif self.method == 'soft_min_distance':
            # b -> [batch_size, centroid_size, input_window_size, input_size]
            b = x.repeat(self.shapelet_size, 1, 1, 1).permute(1, 0, 2, 3)

            # c -> [centroid_size, input_window_size, input_size]
            c = self.centroids.repeat(self.input_size, 1, 1).permute(1, 2, 0)

            # distances -> [batch_size, centroid_size, input_size]
            distances = ((b - c) ** 2).mean(dim=2)

            # soft_min_distance -> [batch_size, centroid_size, input_size]
            soft_min = self.softmin(distances)

            # soft_min_distance -> [batch_size, input_size, centroid_size]
            soft_min = soft_min.permute(0, 2, 1)

            # ret -> [batch_size, input_size, output_window_size]
            ret = self.l1(soft_min)

            ret = ret.permute(0, 2, 1)
            # ret = torch.relu(ret)

            return ret