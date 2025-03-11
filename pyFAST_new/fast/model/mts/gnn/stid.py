#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    """ Multi-Layer Perceptron with residual links. """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
            input data with shape [B, D, N]
        """
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden


class STID(nn.Module):
    """
        Zezhi Shao, Zhao Zhang, Fei Wang, Wei Wei, Yongjun Xu
        Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting[C]//
        Proceedings of the 31st ACM International Conference on Information & Knowledge Management (CIKM),
        ACM, 2022: 4454-4458.

        url: https://doi.org/10.1145/3511808.3557702
        Official Code: https://github.com/zezhishao/STID

        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param node_dim: spatial node embedding dimensionality.
        :param embed_dim: temporal sequence embedding dimensionality.
        :param input_dim: input feature dimensionality (1 for univariate).
        :param num_layer: number of MLP layers in encoder.
        :param if_node: whether to enables spatial identity mechanism.
    """

    def __init__(self, input_window_size: int, output_window_size: int, output_vars: int,
                 node_dim: int = 32, embed_dim: int = 1024, input_dim: int = 1,
                 num_layer: int = 1, if_node: bool = True):
        super().__init__()
        # attributes
        self.num_nodes = output_vars
        self.node_dim = node_dim
        self.input_len = input_window_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.output_len = output_window_size
        self.num_layer = num_layer

        self.if_spatial = if_node

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x -> [batch_size, input_window_size, input_size]"""

        # prepare data
        history_data = x.unsqueeze(-1)  # => [batch_size, input_window_size, input_size, 1]
        input_data = history_data[..., range(self.input_dim)]  # => [batch_size, input_window_size, input_size, 1]

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()  # => [batch_size, input_size, input_window_size, 1]

        # => [batch_size, input_size, input_window_size, 1]
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data.float())  # => [batch_size, embed_dim, input_size, 1]

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            # => [batch_size, embed_dim / batch_size, input_size, 1]
            node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # concatenate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb, dim=1)  # => [batch_size, node_dim + embed_dim, input_size, 1]

        # encoding
        hidden = self.encoder(hidden)  # => [batch_size, node_dim + embed_dim, input_size, 1]

        # regression
        prediction = self.regression_layer(hidden)  # => [batch_size, output_window_size, input_size, 1]
        output = prediction.squeeze(-1)  # => [batch_size, output_window_size, input_size]

        return output
