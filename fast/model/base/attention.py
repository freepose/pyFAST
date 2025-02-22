#!/usr/bin/env python
# encoding: utf-8
import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
        Self-attention mechanism.
    """

    def __init__(self, input_vars: int, method: str = 'projection'):
        super(SelfAttention, self).__init__()
        self.input_vars = input_vars
        self.method = method

        self.projection = nn.Sequential(
            nn.Linear(self.input_vars, self.input_vars // 2),
            nn.ReLU(True),
            nn.Linear(self.input_vars // 2, 1)
        )

    def forward(self, inputs: torch.tensor):
        """
            :param inputs: shape [batch_size, window_size, input_vars]
            :return: outputs [batch_size, window_size, input_vars]
        """
        energy = self.projection(inputs)  # => [batch_size, window_size, 1]
        weight = energy.softmax(dim=1)  # => [batch_size, window_size, input_vars(?)]

        # [batch_size, window_size, input_vars] * [batch_size, window_size, 1] -> [batch_size, window_size]
        context = (inputs * weight).sum(dim=1)

        return weight, context


class SymmetricAttention(nn.Module):
    def __init__(self, input_vars: int, hidden_size: int = 32, dim: int = 1):
        super().__init__()
        self.input_vars = input_vars
        self.hidden_size = hidden_size
        self.dim = dim
        self.Ml = nn.Linear(input_vars, hidden_size, bias=False)
        self.Mr = nn.Linear(hidden_size, input_vars, bias=False)

    def forward(self, queries):
        attention = self.Ml(queries)  # [batch_size, window_size, hidden_size]
        if self.dim > -1:
            attention = attention.softmax(self.dim)  # attention => [batch_size, window_size, hidden_size]
        out = self.Mr(attention)  # out => [batch_size, window_size, input_vars]
        return out


class ExternalAttention(nn.Module):
    """
        Lv, J., Zhang, H., & Liu, S. (2020).
        Beyond self-attention: External attention using two linear layers for visual tasks.
        In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
        (pp. 12354-12363).
    """

    def __init__(self, num_channels):
        super(ExternalAttention, self).__init__()
        self.query_layer = nn.Linear(num_channels, num_channels)
        self.key_layer = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        query = torch.tanh(self.query_layer(x))
        key = torch.tanh(self.key_layer(x))
        attention_weight = torch.matmul(query, key.T) / math.sqrt(key.shape[1])
        return attention_weight


class MultiHeadSymmetricAttention(nn.Module):
    """
        Meng-Hao Guo and Zheng-Ning Liu and Tai-Jiang Mu and Shi-Min Hu,
        Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks,
        preprint arXiv:2105.02358, 2021.
        doi: arXiv:2105.02358, URL: https://arxiv.org/abs/2105.02358

    """

    def __init__(self, input_vars: int, d_model: int = 16, num_heads: int = 2, hidden_size: int = 16,
                 dropout_rate: float = 0., normalized: bool = False):
        super().__init__()
        self.input_vars = input_vars
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.d_model = d_model
        self.normalized = normalized

        self.linear_align = nn.Linear(self.input_vars, self.d_model, bias=False)

        self.Ml = nn.Linear(self.d_model // self.num_heads, self.hidden_size, bias=False)
        self.Mr = nn.Linear(hidden_size, self.d_model // self.num_heads, bias=False)
        self.Wo = nn.Linear(self.d_model, self.input_vars, bias=False)

        nn.init.xavier_uniform_(self.linear_align.weight, gain=0.05)
        nn.init.xavier_uniform_(self.Ml.weight, gain=0.05)
        nn.init.xavier_uniform_(self.Mr.weight, gain=0.05)
        nn.init.xavier_uniform_(self.Wo.weight, gain=0.05)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        # alignment-function
        queries = self.linear_align(inputs)  # => [batch_size, window_size, d_model]
        # => [batch_size, window_size, num_heads, d_model//num_heads]
        queries = queries.view(inputs.shape[0], inputs.shape[1], self.num_heads, self.d_model // self.num_heads)
        queries = queries.permute(0, 2, 1, 3)  # => [batch_size, num_heads, window_size, d_model//num_heads]
        attention = self.Ml(queries)  # => [batch_size, num_heads，window_size, hidden_size]
        attention = attention.softmax(2)  # => [batch_size, num_heads，window_size, hidden_size]
        if self.normalized:
            attention = attention / attention.sum(dim=3,
                                                  keepdim=True)  # => [batch_size, num_heads，window_size, hidden_size]
        out = self.Mr(attention)  # => [batch_size, num_heads, window_size, d_model//num_heads]
        out = out.permute(0, 2, 1, 3).reshape(inputs.shape[0], inputs.shape[1], -1)
        out = self.dropout(self.Wo(out))
        return out
