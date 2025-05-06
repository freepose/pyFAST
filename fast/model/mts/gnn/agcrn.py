#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class AVWGCN(nn.Module):
    """ Adaptive Vertex-weighted Graph Convolution """

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super(AVWGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor):
        """
            :param x: (B, N, C)
            :param node_embeddings: (N, D), supports shaped (N, N)
            :return: (B, N, C)
        """
        # x
        node_num = node_embeddings.shape[0]
        supports = torch.softmax(torch.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        # N, cheb_k, dim_in, dim_out
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    """ Adaptive Gated RNN Cell """

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 *
                           dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim,
                             dim_out, cheb_k, embed_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor):
        """
            :param x: (B, num_nodes, input_dim)
            :param state: (B, num_nodes, hidden_dim)
        """
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)  # (B, num_nodes, input_dim+hidden_dim)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    """ Adaptive Spatiotemporal DCRNN """

    def __init__(self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int, num_layers: int):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x: torch.Tensor, init_state: torch.Tensor, node_embeddings: torch.Tensor):
        """
            :param x: (B, T, N, D)
            :param init_state: (num_layers, B, N, hidden_dim)
            :param node_embeddings: (D, embed_dim)
        """
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]  # seq_length: T
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]  # (B, N, hidden_dim)
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.dcrnn_cells[i].init_hidden_state(batch_size))

        return torch.stack(init_states, dim=0)  # return -> (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    """
        Lei Bai, Lina Yao, Can Li, Xianzhi Wang, Can Wang
        Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting
        url: https://arxiv.org/abs/2007.02842

        Official Code: https://github.com/LeiBAI/AGCRN

        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param input_dim: the input feature dimension of each node.
        :param output_dim: the output feature dimension of each node.
        :param rnn_units: the hides layer dimensions of GRU unit.
        :param num_layers: the number of stacking layers
        :param default_graph: True for static graphs; False for dynamic graphs。
        :param embed_dim: node embedding dimension, controlling the size of the feature space for adaptive parameter learning of nodes.
        :param cheb_k: the order of Chebyshev polynomials determines the spatiotemporal receptive field range of graph convolution.
    """

    def __init__(self, input_vars: int = 1, output_window_size: int = 1, input_dim: int = 1, output_dim: int = 1,
                 rnn_units: int = 32, num_layers: int = 2, default_graph: bool = True,
                 embed_dim: int = 3, cheb_k: int = 2):
        super(AGCRN, self).__init__()

        self.input_size = input_vars
        self.output_window_size = output_window_size

        self.horizon = self.output_window_size
        self.num_node = self.input_size

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = rnn_units
        self.num_layers = num_layers

        self.default_graph = default_graph
        self.node_embeddings = nn.Parameter(torch.randn(
            self.num_node, embed_dim), requires_grad=True)

        self.encoder = AVWDCRNN(self.num_node, self.input_dim, self.hidden_dim, cheb_k,
                                embed_dim, self.num_layers)

        # predictor
        self.end_conv = nn.Conv2d(1, self.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_param()

    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            :param x: (batch_size, input_window_size, input_vars)
        """
        x = x.unsqueeze(-1)  # -> [batch_size, input_window_size, input_size, 1]
        init_state = self.encoder.init_hidden(x.shape[0])  # -> [num_layers, batch_size, input_size, hidden_dim]
        output, _ = self.encoder(
            x, init_state, self.node_embeddings)  # output -> [batch_size, input_window_size, input_size, hidden_dim]
        output = output[:, -1:, :, :]  # output -> [batch_size, 1, input_size, hidden_dim]

        # CNN based predictor
        output = self.end_conv(output)  # -> [batch_size, output_window_size(horizon) * output_dim, input_size, 1]
        output = output.squeeze(-1)  # -> [batch_size, output_window_size(horizon) * output_dim, input_size(num_node)]
        output = output.reshape(-1, self.horizon, self.output_dim,
                                self.num_node)  # -> [batch_size, output_window_size(horizon), output_dim, input_size(num_node)]
        output = output.permute(0, 1, 3, 2)  # -> [batch_size, output_window_size(horizon), input_size, output_dim(1)]

        # add: reduce output dimension
        output = output.squeeze(-1)  # -> [batch_size, output_window_size(horizon), input_size]

        return output
