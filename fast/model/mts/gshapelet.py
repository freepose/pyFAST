#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal, Tuple, List
from .ar import AR
from ...data import PatchMaker

DistanceType = Literal['min_euclidean', 'max_product', 'max_cosine', 'max_cross_correlation']


class ShapeletBlock(nn.Module):
    """
        Shapelet block for high-dimensional time series.
    """

    def __init__(self,
                 n_vars: int,
                 shapelet_len: int,
                 shapelet_num: int,
                 d_model: int = 64,
                 distance: DistanceType = 'min_euclidean'):
        super(ShapeletBlock, self).__init__()

        self.n_vars = n_vars
        self.shapelet_len = shapelet_len
        self.shapelet_num = shapelet_num
        self.d_model = d_model
        self.distance = distance

        self.candidates = nn.Parameter(torch.randn(n_vars, shapelet_num, shapelet_len) * 0.01)
        self.fc = AR(self.shapelet_num, self.n_vars, self.d_model)  # projection for each variable

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: Input tensor of shape [batch_size, input_window_size, n_vars],
                        subject to ``shapelet_len <= input_window_size``.
            :param x_mask: Mask tensor of shape [batch_size, input_window_size, n_vars]

            :return: Output tensor of shape [batch_size, hidden_size, n_vars]
        """

        batch_size, input_window_size, n_vars = x.shape
        patch_maker = PatchMaker(seq_len=input_window_size, patch_len=self.shapelet_len)
        x_segments = patch_maker(x)  # -> (batch_size, n_vars, n_patches, shapelet_len)
        if x_mask is not None:
            mask_segments = patch_maker(x_mask)

        if self.distance == 'min_euclidean':
            # the Euclidean distance between each segment and each shapelet
            diff = (x_segments.unsqueeze(3) - self.candidates.unsqueeze(0).unsqueeze(2)) ** 2  # -> [b, v, p, s, l]

            if x_mask is not None:
                diff = diff * mask_segments.unsqueeze(3)

            dist = diff.sum(dim=4)  # sum over length -> [b, v, p, s]
            dist, _ = dist.min(dim=2)  # min over patches -> [b, v, s]

            dist = dist.permute(0, 2, 1)  # -> [b, s, v]
            dist = dist.softmax(dim=1)
            out = self.fc(dist)  # -> [b, hidden_size, v]

            return out

        elif self.distance == 'max_product':
            if x_mask is not None:
                x_segments[~mask_segments] = 0.

            prod = torch.einsum('b v p l, v s l -> b v p s', x_segments, self.candidates)
            prod, _ = prod.max(dim=2)  # max over shapelets -> (batch_size, n_vars, shapelet_num)

            prod = prod.permute(0, 2, 1)  # -> [b, s, v]
            prod = prod.softmax(dim=1)
            out = self.fc(prod)  # -> [b, hidden_size, v]

            return out
        elif self.distance == 'max_cosine':
            if x_mask is not None:
                x_segments[~mask_segments] = 0.0

            x_norm = F.normalize(x_segments, dim=-1)
            cand_norm = F.normalize(self.candidates, dim=-1)

            cos = torch.einsum('b v p l, v s l -> b v p s', x_norm, cand_norm)
            cos, _ = cos.max(dim=2)  # max over shapelets -> (batch_size, n_vars, shapelet_num)

            cos = cos.permute(0, 2, 1)  # -> [b, s, v]
            cos = cos.softmax(dim=1)
            out = self.fc(cos)  # -> [b, hidden_size, v]

            return out
        elif self.distance == 'max_cross_correlation':
            # center with mask
            if x_mask is not None:
                valid_len = mask_segments.sum(dim=-1, keepdim=True) + 1e-6
                x_centered = (x_segments * mask_segments) - \
                             (x_segments * mask_segments).sum(dim=-1, keepdim=True) / valid_len
            else:
                x_centered = x_segments - x_segments.mean(dim=-1, keepdim=True)

            cand_centered = self.candidates - self.candidates.mean(dim=-1, keepdim=True)
            x_norm = F.normalize(x_centered, dim=-1)
            cand_norm = F.normalize(cand_centered, dim=-1)

            corr = torch.einsum('b v p l, v s l -> b v p s', x_norm, cand_norm)
            corr, _ = corr.max(dim=2)  # max over shapelets -> (batch_size, n_vars, shapelet_num)
            corr = corr.permute(0, 2, 1)  # -> [b, s, v]
            corr = corr.softmax(dim=1)
            out = self.fc(corr)  # -> [b, hidden_size, v]
            return out

        else:
            raise ValueError(f"Unknown distance type: {self.distance}")


class ShapeletLayer(nn.Module):
    def __init__(self,
                 n_vars: int,
                 shapelets_len_and_num: List[Tuple[int, int]],
                 d_model: int = 64,
                 distance: DistanceType = 'min_euclidean'):
        super(ShapeletLayer, self).__init__()

        self.n_vars = n_vars
        self.shapelets_len_and_num = shapelets_len_and_num
        self.d_model = d_model
        self.distance = distance

        self.blocks = nn.ModuleList([
            ShapeletBlock(n_vars=self.n_vars, shapelet_len=shapelet_len, shapelet_num=shapelet_num,
                          d_model=self.d_model, distance=self.distance)
            for i, (shapelet_len, shapelet_num) in enumerate(shapelets_len_and_num)
        ])

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        """
            :param x: input window, [batch_size, window_size, n_vars]
            :param x_mask: mask of input window [batch_size, window_size, n_vars]
            :return: [batch_size, shapelet_num]
        """
        blocks_out = []
        for block in self.blocks:
            blocks_out.append(block(x, x_mask))

        out = torch.cat(blocks_out, dim=1)  # -> [batch_size, total_shapelet_block * hidden_size, n_vars]
        return out


class GShapelets(nn.Module):

    def __init__(self, input_window_size: int, input_vars: int,
                 output_window_size: int,
                 shapelets_len_and_num: List[Tuple[int, int]],
                 d_model: int = 64,
                 distance: DistanceType = 'min_euclidean',
                 dropout_rate: float = 0.):
        super(GShapelets, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.d_model = d_model
        self.distance = distance

        self.shapelets_len_and_num = shapelets_len_and_num

        self.shapelet_layer = ShapeletLayer(n_vars=input_vars,
                                            shapelets_len_and_num=self.shapelets_len_and_num,
                                            d_model=self.d_model,
                                            distance=distance)

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(len(shapelets_len_and_num) * self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, output_window_size)
        )

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: input window, [batch_size, window_size, n_vars]
            :param x_mask: mask of input window [batch_size, window_size, n_vars]

            :return: [batch_size, output_window_size, output_vars]
        """

        features = self.shapelet_layer(x, x_mask)  # -> [batch_size, total_shapelet_blocks * d_model, n_vars]

        features = features.permute(0, 2, 1)  # -> [batch_size, n_vars, total_shapelet_blocks * d_model]
        out = self.fc(features)  # -> [batch_size, n_vars, output_window_size]
        out = out.permute(0, 2, 1)  # -> [batch_size, output_window_size, n_vars]

        out = self.dropout(out)

        return out
