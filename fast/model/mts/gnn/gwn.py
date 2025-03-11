#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn
import torch.nn.functional as F


class nconv(nn.Module):
    """ Graph conv operation. """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
            :param x: [batch_size, in_channels, num_nodes, num_nodes]
            :param A: [num_nodes, num_nodes]
        """
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    """ Linear layer. """
    def __init__(self, c_in: int, c_out: int):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class gcn(nn.Module):
    """ Graph convolution network. """
    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int = 3, order: int = 2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x: torch.Tensor, support: list[torch.Tensor]):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a.to(x.device))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x.device))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GraphWaveNet(nn.Module):
    """
        Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang
        Graph WaveNet for Deep Spatial-Temporal Graph Modeling
        Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI).
        2019: 1907-1913

        url: https://doi.org/10.48550/arXiv.1906.00121
        code: https://github.com/GestaltCogTeam/BasicTS/blob/master/baselines/GWNet/arch/gwnet_arch.py

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param in_dim: input dimensionality.
        :param out_dim: output dimensionality.
        :param supports: list of adjacency matrices for graph convolutions.
        :param gcn_bool: whether to use GCN layers.
        :param addaptadj: whether to enable adaptive adjacency matrix.
        :param aptinit: pre-trained node embedding matrix for adaptive adj.
        :param residual_channels: residual connection channel count.
        :param dilation_channels: dilation convolution channel count.
        :param skip_channels: skip connection channel count.
        :param end_channels: final layer channel count.
        :param kernel_size: dilation kernel size.
        :param blocks: number of dilation blocks.
        :param layers: layers per dilation block.
        :param dropout: dropout rate.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 in_dim: int = 1, out_dim: int = 1, supports: list[torch.Tensor] = None, gcn_bool: bool = True,
                 addaptadj: bool = True, aptinit: torch.Tensor = None, residual_channels: int = 32,
                 dilation_channels: int = 32, skip_channels: int = 256, end_channels: int = 512, kernel_size: int = 2,
                 blocks: int = 4, layers: int = 2, dropout: float = 0.3):
        super(GraphWaveNet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(
                    torch.randn(input_vars, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(
                    torch.randn(10, input_vars), requires_grad=True)
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.temporal_projector = nn.Linear(self.input_window_size - 3 * blocks, self.output_window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x => [batch_size, input_window_size, input_size] """
        x = x.unsqueeze(-1)  # -> [batch_size, input_window_size, input_size, 1]

        input = x.transpose(1, 3).contiguous()  # -> [batch_size, 1, input_size, input_window_size]
        in_len = input.size(3)  # -> input_window_size
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input  # -> [batch_size, 1, input_size, input_window_size]
        x = self.start_conv(
            x)  # -> [batch_size, residual_channels(start_conv:out_channels), input_size, input_window_size]
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # -> [input_size, input_size]
            new_supports = self.supports + [adp]  # new_supports(list) -> {support:None; adp:[input_size, input_size]}

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            # (dilation, init_dilation) = self.dilations[i]

            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x  # residual -> [batch_size, residual_channels(start_conv_out_channels), input_size, 'T'] T会一直变小，根据(T-kernel)/1 + 1
            # dilated convolution
            filter = self.filter_convs[i](
                residual)  # filter -> [batch_size, dilation_channels(filter_convs:out_channels), input_size, (input_window_size-kernel)/1 + 1]
            filter = torch.tanh(
                filter)  # filter -> [batch_size, dilation_channels, input_size, (input_window_size-kernel)/1 + 1]
            gate = self.gate_convs[i](
                residual)  # gate -> [batch_size, dilation_channels(gate_convs:out_channels), input_size, (input_window_size-kernel)/1 + 1]
            gate = torch.sigmoid(
                gate)  # gate -> [batch_size, dilation_channels, input_size, (input_window_size-kernel)/1 + 1]
            x = filter * gate  # x -> [batch_size, dilation_channels, input_size, (input_window_size-kernel)/1 + 1]

            # parametrized skip connection
            s = x  # s -> [batch_size, dilation_channels, input_size, (input_window_size-kernel)/1 + 1]
            s = self.skip_convs[i](
                s)  # s -> [batch_size, skip_channels(skip_channel:out_channels), input_size, (input_window_size-kernel)/1 + 1]
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip  # s -> [batch_size, skip_channels(skip_channel:out_channels), input_size, (input_window_size-kernel)/1 + 1]

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x,
                                      new_supports)  # x -> [batch_size, skip_channel, input_size, (input_window_size-kernel)/1 + 1]
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            # x -> [batch_size, skip_channel, input_size, (input_window_size-kernel)/1 + 1]
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)  # x -> [batch_size, skip_channel, input_size, (input_window_size-kernel)/1 + 1]

        x = torch.relu(skip)
        x = torch.relu(self.end_conv_1(x))  # x -> [batch_size, end_channels, input_size, input_window_size - block * 3]
        x = self.end_conv_2(x)  # x -> [batch_size, out_dim, input_size, input_window_size - block * 3]
        x = self.temporal_projector(x)  # x -> [batch_size, out_dim, input_size, output_window_size]
        x = x.squeeze(1).transpose(1, 2).contiguous()  # x -> [batch_size, output_window_size, input_size]

        return x
