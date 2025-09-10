#!/usr/bin/env python
# encoding: utf-8

from typing import Literal

import torch
import torch.nn as nn

from fast.data import PatchMaker
from fast.model.base import MLP
from fast.model.mts.transformer.embedding import PositionalEncoding


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x (B, F, N, M)
        # A (B, M, N, N)
        x = torch.einsum('bfnm,bmnv->bfvm', (x, A))  # -> (B, F, N, M)
        return x  # (B, F, N, M)


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # self.mlp = nn.Linear(c_in, c_out)
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        # x (B, F, N, M)
        out = self.mlp(x)  # -> (B, F', N, M)
        return out


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        # c_in = (order*support_len)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        # x (B, F, N, M)
        # a (B, M, N, N)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)  # concat x and x_conv
        h = self.mlp(h)  # -> (B, F', N, M)
        return torch.relu(h)


class TPatchGNN(nn.Module):
    """
        Weijia Zhang, Chenlong Yin, Hao Liu, Xiaofeng Zhou, Hui Xiong.
        Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach.
        ICML 2024

        The author code is available at: https://github.com/usail-hkust/t-PatchGNN

        Time series are typically assumed to be generated at regularly spaced interval of time,
        and so are called **regular time series**. The data can include a timestamp explicitly or a timestamp
        can be implied based on the intervals at which the data is created.
        Time series without an associated timestamp are automatically assumed to be regular time series.

        An **irregular time series** is the opposite of a regular time series. The data in the time series
        follows a temporal sequence, but the measurements might not happen at a regular time interval.
        For example, the data might be generated as a burst or with varying time intervals.
        Account deposits or withdrawals from an ATM are examples of an irregular time series.

        -- from https://www.ibm.com/docs/en/streams/4.3.0?topic=series-regular-irregular-time

        :param input_window_size: input window size
        :param input_vars: input variables of target time series, is the same as ``output_vars``.
        :param output_window_size: output window size.
        :param output_vars: output variables of target time series, is the same as ``input_vars``.
        :param ex_vars: the number of exogenous variables.
        :param ex2_vars: the number of second exogenous variables for the future time steps.
        :param patch_len: the length of the patch.
        :param patch_stride: the stride of the patch.
        :param time_embedding_dim: the dimension of the time embedding.
        :param hidden_dim: the hidden dimension for the intra-time series modeling.
        :param num_layers: the number of layers for the intra-time series modeling.
        :param transformer_nhead: the number of heads in the transformer.
        :param transformer_num_layers: the number of layers in the transformer.
        :param node_vector_dim: the dimension of the node vector.
        :param dropout_rate: the dropout rate.
        :param supports: the adjacency matrix if GCN.
        :param gcn_hop: the hop of the GCN, also used as GCN order.
        :param aggregation: the aggregation method, 'linear' or 'cnn'.
    """

    def __init__(self, input_window_size: int, input_vars: int,
                 output_window_size: int, output_vars: int,
                 ex_vars: int, ex2_vars: int,
                 patch_len: int = None, patch_stride: int = None,
                 time_embedding_dim: int = 20,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 transformer_nhead: int = 2,
                 transformer_num_layers: int = 2,
                 node_vector_dim: int = 10,
                 dropout_rate: float = 0.,
                 supports: list = None,
                 gcn_hop: int = 1,
                 aggregation: Literal['linear', 'cnn'] = 'cnn'):

        super(TPatchGNN, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.ex_vars = ex_vars
        self.ex2_vars = ex2_vars

        self.time_embedding_dim = time_embedding_dim
        self.hidden_dim = hidden_dim  # Hidden dimension for the intra-time series modeling
        self.transformer_nhead = transformer_nhead
        self.transformer_num_layers = transformer_num_layers

        self.num_layers = num_layers  # Number of layers for the intra-time series modeling

        self.node_vector_dim = node_vector_dim
        self.aggregation = aggregation

        self.patch_len = patch_len if patch_len is not None else input_window_size
        self.patch_stride = patch_stride if patch_stride is not None else input_window_size
        assert self.patch_len <= self.input_window_size, 'The patch length must be less than or equal to the input window size.'

        self.patch_maker = PatchMaker(input_window_size, self.patch_len, self.patch_stride)
        self.patch_num = self.patch_maker.patch_num

        """
            Intra-time series modeling, a.k.a., model with several time steps
        """
        # Continuous time embedding
        self.time_embedding_scale = nn.Linear(1, 1)
        self.time_embedding_periodic = nn.Linear(1, time_embedding_dim - 1)

        filter_input_dim = self.time_embedding_dim + 1
        self.ttcn_dim = self.hidden_dim - 1

        self.filter_generators = MLP(filter_input_dim, [self.ttcn_dim, self.ttcn_dim], filter_input_dim * self.ttcn_dim,
                                     None, 'relu')
        self.T_bias = nn.Parameter(torch.randn(1, self.ttcn_dim), requires_grad=True)

        """
            Inter-time series modeling, a.k.a., model with several variables. 
        """
        self.d_model = self.hidden_dim  # d_model is the hidden dimension of the transformer
        self.positional_encoding = PositionalEncoding(self.d_model, self.input_window_size)

        # Time-adaptive Graph Structure Learning
        self.node_vector1 = nn.Parameter(torch.randn(self.input_vars, self.node_vector_dim), requires_grad=True)
        self.node_vector2 = nn.Parameter(torch.randn(self.node_vector_dim, self.input_vars), requires_grad=True)

        self.transformer_encoder = nn.ModuleList()
        self.node_vec_linear1 = nn.ModuleList()
        self.node_vec_linear2 = nn.ModuleList()
        self.node_vec_gate1 = nn.ModuleList()
        self.node_vec_gate2 = nn.ModuleList()

        self.supports = [] if supports is None else supports
        self.support_len = len(self.supports) + 1  # Default as 1
        self.gcn_hop = gcn_hop
        self.gcn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.transformer_nhead, batch_first=True)
            self.transformer_encoder.append(nn.TransformerEncoder(encoder_layer, self.transformer_num_layers))

            self.node_vec_linear1.append(nn.Linear(self.hidden_dim, self.node_vector_dim))
            self.node_vec_linear2.append(nn.Linear(self.hidden_dim, self.node_vector_dim))

            self.node_vec_gate1.append(nn.Sequential(
                nn.Linear(self.hidden_dim + self.node_vector_dim, 1),
                nn.Tanh(),
                nn.ReLU()))
            self.node_vec_gate2.append(nn.Sequential(
                nn.Linear(self.hidden_dim + self.node_vector_dim, 1),
                nn.Tanh(),
                nn.ReLU()))

            self.gcn_layers.append(gcn(self.d_model, self.d_model, dropout_rate,
                                       support_len=self.support_len, order=self.gcn_hop))

        encoder_dim = self.hidden_dim

        if self.aggregation == 'linear':
            self.temporal_agg = nn.Sequential(nn.Linear(self.hidden_dim * self.patch_num, encoder_dim))
        else:
            self.temporal_agg = nn.Sequential(nn.Conv1d(self.d_model, encoder_dim, kernel_size=self.patch_num))

        self.ex2_linear1 = nn.Linear(self.ex2_vars, 1)

        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim + self.time_embedding_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1)
        )

    def continuous_time_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
            Continuous time embedding
            :param x: shape is (..., 1)
            :return: shape is (..., time_embedding_dim)
        """

        x1 = self.time_embedding_scale(x)  # -> (..., 1)
        x2 = torch.sin(self.time_embedding_periodic(x))  # -> (..., time_embedding_dim - 1)
        out = torch.cat([x1, x2], dim=-1)

        return out

    def transformable_time_aware_convolution(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
            Transformable time-aware convolution (TTCN).
            :param x: shape is (batch_size * input_vars * patch_num, patch_len, 1 + te_dim)
            :param x_mask: shape is (batch_size * input_vars * patch_num, patch_len, 1)
            :return:
        """
        x_filter = self.filter_generators(x)  # -> (bs * input_vars * patch_num, patch_len, (te_dim + 1) * ttcn_dim)
        x_filter_mask = x_filter * x_mask.float() + (1 - x_mask.float()) * (-1e8)

        x_filter_norm = x_filter_mask.softmax(dim=1)  # Normalization on sequence dimension
        x_filter_norm = x_filter_norm.unflatten(2, [self.ttcn_dim, -1])
        # -> (bs * input_vars * patch_num, patch_len, ttcn_dim, te_dim + 1)

        x_broadcast = x.unsqueeze(2).repeat(1, 1, self.ttcn_dim, 1)
        # -> (bs * input_vars * patch_num, patch_len, ttcn_dim, te_dim + 1)

        out = (x_broadcast * x_filter_norm).sum(dim=1).sum(2)  # -> (bs * input_vars * patch_num, ttcn_dim)
        out = out + self.T_bias  # -> (bs * input_vars * patch_num, ttcn_dim)
        out = torch.relu(out)

        return out

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, ex2: torch.Tensor):
        """

            :param x: shape is (batch_size, input_window_size, input_vars)
            :param x_mask: shape is the same as ``x``
            :param ex2: shape is (batch_size, input_window_size + output_window_size, ex2_vars).
                        The pre-known exogenous variables, e.g., time.
            :return: shape is (batch_size, output_window_size, output_vars).
        """
        if x_mask is not None:
            x[~x_mask] = 0.

        x = self.patch_maker(x)  # -> (batch_size, input_vars, patch_num, patch_len)
        x_mask = self.patch_maker(x_mask)  # -> (batch_size, input_vars, patch_num, patch_len)

        batch_size, input_vars, patch_num, patch_len = x.shape

        x = x.flatten(0, 2).unsqueeze(-1)  # -> (batch_size * input_vars * patch_num, patch_len, 1)
        x_mask = x_mask.flatten(0, 2).unsqueeze(-1)  # -> (batch_size * input_vars * patch_num, patch_len, 1)

        x_time_emb = self.continuous_time_embedding(x)  # -> (batch_size * input_vars * patch_num, te_dim)

        x = torch.cat([x, x_time_emb], dim=-1)  # -> (batch_size * input_vars * patch_num, patch_len, 1 + te_dim)
        hidden = self.transformable_time_aware_convolution(x, x_mask)  # -> (batch_size * input_vars * patch_num, ttcn_dim)

        # Mask for the patch, **ttcn_dim + 1 == hidden_dim**
        mask_patch = (x_mask.float().sum(dim=1) > 0)  # -> (batch_size * input_vars * patch_num, 1)
        x = torch.cat([hidden, mask_patch], dim=-1)  # -> (batch_size * input_vars * patch_num, ttcn_dim + 1)

        x = x.view(batch_size, input_vars, patch_num, -1)  # -> (batch_size, input_vars, patch_num, ttcn_dim + 1)

        for i in range(self.num_layers):
            x_3d = x.flatten(0, 1)  # -> (batch_size * input_vars, patch_num, ttcn_dim + 1)
            pe = self.positional_encoding(x_3d)  # -> (batch_size * input_vars, patch_num, ttcn_dim + 1)
            x_3d = x_3d + pe  # -> (batch_size * input_vars, patch_num, ttcn_dim + 1)

            x_3d = self.transformer_encoder[i](x_3d)  # -> (batch_size * input_vars, patch_num, ttcn_dim + 1)
            x = x_3d.unflatten(0, (batch_size, input_vars))  # -> (batch_size, input_vars, patch_num, ttcn_dim + 1)

            node_vec1 = self.node_vector1.unsqueeze(0).unsqueeze(0).repeat(batch_size, patch_num, 1, 1)
            node_vec2 = self.node_vector2.unsqueeze(0).unsqueeze(0).repeat(batch_size, patch_num, 1, 1)

            x1 = torch.cat([x, node_vec1.permute(0, 2, 1, 3)], dim=3)
            x2 = torch.cat([x, node_vec2.permute(0, 3, 1, 2)], dim=3)
            x_gate1 = self.node_vec_gate1[i](x1)  # -> (batch_size, input_vars, patch_num, 1)
            x_gate2 = self.node_vec_gate2[i](x2)  # -> (batch_size, input_vars, patch_num, 1)

            x_p1 = x_gate1 * self.node_vec_linear1[i](x)  # -> (batch_size, input_vars, patch_num, node_vector_dim)
            x_p2 = x_gate2 * self.node_vec_linear2[i](x)  # -> (batch_size, input_vars, patch_num, node_vector_dim)

            node_vec1 = node_vec1 + x_p1.permute(0, 2, 1, 3)  # -> (batch_size, patch_num, input_vars, node_vector_dim)
            node_vec2 = node_vec2 + x_p2.permute(0, 2, 3, 1)  # -> (batch_size, patch_num, node_vector_dim, input_vars)

            product = torch.matmul(node_vec1, node_vec2)  # -> (batch_size, patch_num, input_vars, input_vars)
            product = torch.relu(product)
            product = product.softmax(dim=3)

            supports = self.supports + [product]  # Adjacency matrix
            x = self.gcn_layers[i](x.permute(0, 3, 1, 2), supports)  # -> (bs, hidden_dim, input_vars, patch_num)

            x = x.permute(0, 2, 3, 1)  # -> (batch_size, input_vars, patch_num, hidden_dim)

        if self.aggregation == 'linear':
            h = x.flatten(2, 3)  # -> (batch_size, input_vars, hidden_dim * patch_num)
            h = self.temporal_agg(h)  # -> (batch_size, input_vars, hidden_dim)
        else:
            h = x.flatten(0, 1).permute(0, 2, 1)  # -> (batch_size * input_vars, hidden_dim, patch_num)
            h = self.temporal_agg(h).squeeze(2)  # -> (batch_size * input_vars, hidden_dim)
            h = h.unflatten(0, (batch_size, input_vars))  # -> (batch_size, input_vars, hidden_dim)

        # Decoder
        ex2_input_window_size = ex2.shape[1]

        h = h.unsqueeze(2).repeat(1, 1, ex2_input_window_size, 1)
        upcoming_time = ex2  # -> (batch_size, ex2_input_window_size, ex2_vars)

        upcoming_time = upcoming_time.unsqueeze(1)  # -> (batch_size, 1, ex2_input_window_size, ex2_vars)
        upcoming_time = upcoming_time.repeat(1, input_vars, 1,
                                             1)  # -> (batch_size, input_vars, ex2_input_window_size, ex2_vars)

        upcoming_time = self.ex2_linear1(upcoming_time)  # -> (..., 1)
        upcoming_time_embedding = self.continuous_time_embedding(upcoming_time)  # (..., time_embedding_dim)

        ret = torch.cat([h, upcoming_time_embedding], dim=3)  # -> (..., hidden_dim + time_embedding_dim)

        ret = self.decoder(ret)  # -> (batch_size, input_vars, ex2_input_window_size, 1)
        ret = ret.squeeze(3).permute(0, 2, 1)  # -> (batch_size, ex2_input_window_size, input_vars)

        ret = ret[:, -self.output_window_size:, :]

        return ret
