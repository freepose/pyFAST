#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import numbers
from typing import List, Tuple, Optional, Union


class MAGNet(nn.Module):
    """
        Zonglei Chen, Fan Zhang, Tianrui Li, Chongshou Li
        MAGNet: Multi-scale Attention and Evolutionary Graph Structure for Long Sequence Time-Series Forecasting
        url: https://link.springer.com/chapter/10.1007/978-3-031-44223-0_18
        official code: https://github.com/Masterleia/MAGNet/blob/master/models/MAGNet.py

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param label_window_size: label sequence length of the target time series, typically aligned with the input window.
        :param conv2d_in_channels: input channel count of the 2D convolution layer, corresponding to the feature dimensionality.
        :param residual_channels: channel count of residual connections, used to mitigate gradient vanishing.
        :param conv_channels: output channel count of convolutional layers, controlling feature expressive power.
        :param skip_channels: channel count of skip connections, used for cross-layer information fusion.
        :param end_channels: final output channel count of the model, corresponding to the prediction target dimension.
        :param node_dim: embedding dimension of graph nodes, determining the physical meaning expression capability of graph neural networks.
        :param tanhalpha: scaling factor of the hyperbolic tangent function, affecting the non-linear transformation strength of graph node features.
        :param static_feat: static features (e.g., atomic type encodings) used to enhance the physical prior of graph nodes.
        :param dilation_exponential: exponential factor controlling the dilation rate of convolution kernels, affecting model receptive field size.
        :param kernel_size: size of the convolution kernel (fixed height=1, variable width direction), controlling the range of local feature capture.
        :param gcn_depth: number of graph convolutional network layers, controlling graph feature extraction depth.
        :param gcn_true: flag to enable graph convolutional network modules, controlling whether the model uses graph structures for information propagation.
        :param propalpha: weight attenuation parameter in graph propagation, affecting node information transmission stability.
        :param layer_norm_affline: whether to use learnable affine parameters for layer normalization.
        :param buildA_true: flag to enable dynamic graph construction, controlling physical modeling capability of graph structures.
        :param predefined_A: predefined adjacency matrix; if None, dynamically generated via graph construction.
        :param dropout: dropout rate.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 label_window_size: int = 48, conv2d_in_channels: int = 1, residual_channels: int = 32,
                 conv_channels: int = 32, skip_channels: int = 64, end_channels: int = 128, node_dim: int = 40,
                 tanhalpha: float = 3.0, static_feat: Optional[torch.Tensor] = None, dilation_exponential: int = 1,
                 kernel_size: int = 7, gcn_depth: int = 2, gcn_true: bool = False, propalpha: float = 0.05,
                 layer_norm_affline: bool = True, buildA_true: bool = True, predefined_A: Optional[torch.Tensor] = None,
                 dropout: float = 0.3):
        super(MAGNet, self).__init__()

        self.input_window_size = input_window_size
        self.input_size = input_vars
        self.output_window_size = output_window_size
        self.output_size = output_vars
        self.label_window_size = label_window_size

        self.dropout = dropout

        self.conv2d_in_channels = conv2d_in_channels  # default = 1
        self.residual_channels = residual_channels  # default = 32
        self.conv_channels = conv_channels  # default = 32
        self.skip_channels = skip_channels  # default = 64
        self.end_channels = end_channels  # default = 128

        self.num_nodes = self.input_size
        self.subgraph_size = self.input_size
        self.node_dim = node_dim  # default = 40
        self.tanhalpha = tanhalpha  # default = 3
        self.static_feat = static_feat  # default = None

        self.dilation_exponential = dilation_exponential  # default = 1
        self.num_layer = int(self.input_window_size / 6)
        self.kernel_size = kernel_size  # default = 7

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.channel_attention = nn.ModuleList()
        self.spatial_attention = nn.ModuleList()  # 无用

        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.gcn_depth = gcn_depth  # default = 2
        self.gcn_true = gcn_true  # default = False
        self.propalpha = propalpha  # default = 0.05

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        self.norm = nn.ModuleList()
        self.layer_norm_affline = layer_norm_affline  # default = True

        self.buildA_true = buildA_true  # default = True
        self.predefined_A = predefined_A  # default = None

        self.start_conv = nn.Conv2d(in_channels=self.conv2d_in_channels,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        self.gc = GraphConstructor(nnodes=self.num_nodes, k=self.subgraph_size, dim=self.node_dim,
                                   alpha=self.tanhalpha,
                                   seq_len=self.input_window_size,
                                   static_feat=self.static_feat)

        if self.dilation_exponential > 1:
            self.receptive_field = int(
                1 + (self.kernel_size - 1) * (self.dilation_exponential ** self.num_layer - 1) / (
                            self.dilation_exponential - 1))
        else:
            self.receptive_field = self.num_layer * (self.kernel_size - 1) + 1

        if self.input_window_size > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=self.conv2d_in_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window_size), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, self.input_window_size - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=self.conv2d_in_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=self.residual_channels, out_channels=self.skip_channels,
                                   kernel_size=(1, 1), bias=True)

        for i in range(1):
            if self.dilation_exponential > 1:
                rf_size_i = int(1 + i * (self.kernel_size - 1) * (self.dilation_exponential ** self.num_layer - 1) / (
                            self.dilation_exponential - 1))
            else:
                rf_size_i = i * self.num_layer * (self.kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, self.num_layer + 1):
                if self.dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (self.kernel_size - 1) * (self.dilation_exponential ** j - 1) / (
                                self.dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (self.kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))

                self.gate_convs.append(
                    dilated_inception(self.residual_channels, self.conv_channels, dilation_factor=new_dilation))

                self.channel_attention.append(ChannelAttention(self.residual_channels))
                self.spatial_attention.append(SpatialAttention())  # 无用

                if self.input_window_size > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.input_window_size - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=self.conv_channels, out_channels=self.skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                               self.propalpha))
                    self.gconv2.append(mixprop(self.conv_channels, self.residual_channels, self.gcn_depth, self.dropout,
                                               self.propalpha))

                self.residual_convs.append(nn.Conv2d(in_channels=self.conv_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))

                if self.input_window_size > self.receptive_field:
                    self.norm.append(
                        LayerNorm((self.residual_channels, self.num_nodes, self.input_window_size - rf_size_j + 1),
                                  elementwise_affine=self.layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm((self.residual_channels, self.num_nodes, self.receptive_field - rf_size_j + 1),
                                  elementwise_affine=self.layer_norm_affline))

                new_dilation *= self.dilation_exponential

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels, out_channels=self.end_channels,
                                    kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels, out_channels=self.output_window_size,
                                    kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).repeat(self.input_window_size, 1)

    def forward(self, source: torch.Tensor):
        """
            source -> [batch_size, input_window_size, input_size]
        """
        idx = None

        batch_size, input_window_size, _ = source.shape
        assert input_window_size == self.input_window_size, 'input sequence length not equal to preset sequence length'

        if self.label_window_size > input_window_size:
            self.label_window_size = input_window_size

        source = source.unsqueeze(1)  # -> [batch_size, 1, input_window_size, input_size]
        source = source.transpose(2, 3)  # -> [batch_size, 1, input_size, input_window_size]
        if self.input_window_size < self.receptive_field:
            # -> [batch_size, 1, input_size, receptive_field]
            source = nn.functional.pad(source, (self.receptive_field - self.input_window_size, 0, 0, 0))

        x = self.start_conv(source)  # -> [batch_size, residual_channels, input_size, receptive_field]
        skip = self.skip0(nn.functional.dropout(source, self.dropout,
                                                training=self.training))  # -> [batch_size, skip_channels, input_size, 1]

        # 图学习部分
        if self.gcn_true:
            if self.buildA_true:
                if idx is None:
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_A

        for i in range(self.num_layer):
            residual = x  # -> [batch_size, residual_channels, input_size, receptive_field]
            filter = self.filter_convs[i](x)
            filter_chan_atten = self.channel_attention[i](filter) * filter
            filter = torch.tanh(filter_chan_atten)
            gate = self.gate_convs[i](x)
            gate_chan_atten = self.channel_attention[i](gate) * gate
            gate = torch.sigmoid(gate_chan_atten)
            x = filter * gate
            x = torch.nn.functional.dropout(x, self.dropout, training=self.training)

            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            # 图卷积部分
            if self.gcn_true:
                x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(2, 1))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx[0])
            else:
                x = self.norm[i](x, idx[0])

        skip = self.skipE(x) + skip  # -> [batch_size, skip_channels, input_size, 1]
        x = torch.relu(skip)
        x = self.end_conv_1(x)  # -> [batch_size, end_channels, input_size, 1]
        x = torch.relu(x)
        x = self.end_conv_2(x)  # -> [batch_size, output_window_size, input_size, 1]
        x = x.squeeze(3)  # -> [batch_size, output_window_size, input_size]
        return x


class GraphConstructor(nn.Module):
    def __init__(self, nnodes: int, k: int, dim: int, alpha: float = 3.0, seq_len: int = 96,
                 static_feat: Optional[torch.Tensor] = None):
        super(GraphConstructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.gru1 = nn.GRU(batch_first=True, num_layers=2, input_size=dim, hidden_size=dim)
        self.gru2 = nn.GRU(batch_first=True, num_layers=2, input_size=dim, hidden_size=dim)

        self.seq_len = seq_len
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx: torch.Tensor):

        # out_adj = []
        #
        # for i in range(self.seq_len):
        #     if self.static_feat is None:
        #         nodevec1 = self.emb1(idx)
        #         nodevec2 = self.emb2(idx)
        #     else:
        #         nodevec1 = self.static_feat[idx,:]
        #         nodevec2 = nodevec1
        #
        #     nodevec1_gru,_ = self.gru1(torch.unsqueeze(nodevec1,dim=0))
        #     nodevec2_gru,_ = self.gru2(torch.unsqueeze(nodevec2, dim=0))
        #
        #     nodevec1_gru = torch.squeeze(nodevec1_gru,dim=0)
        #     nodevec2_gru = torch.squeeze(nodevec2_gru, dim=0)
        #
        #     # nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        #     # nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))
        #
        #     nodevec1 = torch.tanh(self.alpha * nodevec1_gru)
        #     nodevec2 = torch.tanh(self.alpha * nodevec2_gru)
        #
        #
        #     a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        #     adj = F.relu(torch.tanh(self.alpha*a))
        #     mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        #     mask.fill_(float('0'))
        #     s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        #     mask.scatter_(1,t1,s1.fill_(1))
        #     adj = adj*mask
        #     out_adj.append(adj)
        #
        # out_adj = torch.stack(out_adj).to(self.device)

        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1_gru, _ = self.gru1(nodevec1)
        nodevec2_gru, _ = self.gru2(nodevec2)

        # nodevec1_gru = torch.squeeze(nodevec1_gru, dim=0)
        # nodevec2_gru = torch.squeeze(nodevec2_gru, dim=0)

        # nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        # nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        nodevec1 = torch.tanh(self.alpha * nodevec1_gru)
        nodevec2 = torch.tanh(self.alpha * nodevec2_gru)

        # a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))

        a = torch.bmm(nodevec1, nodevec2.transpose(2, 1)) - torch.bmm(nodevec2, nodevec1.transpose(2, 1))
        adj = torch.relu(torch.tanh(self.alpha * a))

        mask = torch.zeros(idx.size(0), idx.size(1), idx.size(1))  # TODO add
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 2)
        mask.scatter_(2, t1, s1.fill_(1))
        adj = adj * mask
        # out_adj.append(adj)

        return adj

    def fullA(self, idx: torch.Tensor):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = torch.relu(torch.tanh(self.alpha * a))
        return adj


class dilated_inception(nn.Module):
    def __init__(self, cin: int, cout: int, dilation_factor: int = 2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        # self.kernel_set = [1, 2, 5, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input: torch.Tensor):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # a = self.avg_pool(x)
        # b = self.max_pool(x)
        avg_out = self.shared_MLP(self.avg_pool(x))  # self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.shared_MLP(self.max_pool(x))  # self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        # return self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.GELU()

    def forward(self, x: torch.Tensor):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # return self.sigmoid(x)
        return x


class mixprop(nn.Module):
    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float):
        super(mixprop, self).__init__()
        self.nconv = NConv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x: torch.Tensor, adj_3d: torch.Tensor):
        h = x  # (32,32,7,91)
        out = [h]
        adj = adj_3d
        # conv_a = []
        # for adj in adj_3d:
        #     adj = adj + torch.eye(adj.size(0)).to(x.device)  # 7*7
        #     d = adj.sum(1) # 7
        #     a = adj / d.view(-1, 1) # 7*7
        #     conv_a.append(a)
        # conv_a = torch.stack(conv_a).to(x.device)

        adj = adj + torch.eye(adj.size(1)).repeat(adj.size(0), 1, 1).to(x.device)  # 7*7
        d = adj.sum(2)  # 7
        a = adj / d.unsqueeze(-1)  # 7*7

        # conv_a.append(a)

        for i in range(self.gdep):
            # h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        # x = torch.einsum('ncwl,vw->ncvl',(x,A))
        x = torch.einsum('ncwl,avw->ncvl', (x, A))

        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, idx: torch.LongTensor):
        if self.elementwise_affine:
            return torch.nn.functional.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :],
                                                  self.bias[:, idx, :], self.eps)
        else:
            return torch.nn.functional.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
