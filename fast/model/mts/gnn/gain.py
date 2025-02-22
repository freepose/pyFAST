#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from ..ar import GAR


class GAT(nn.Module):
    """
        Zhao H, Wang Y, Duan J, et al.
        Multivariate time-series anomaly detection via graph attention network[C]//
        2020 IEEE International Conference on Data Mining (ICDM). IEEE, 2020: 841-850.
        https://doi.org/10.1109/ICDM50108.2020.00093
    """

    def __init__(self, in_features: int, out_features: int, n_head: int = 8, alpha: float = 0.2,
                 last: bool = False, dropout_rate: float = 0.2):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.last = last

        Ws = []
        As = []
        for _ in range(self.n_head):
            W = nn.Linear(in_features, out_features)
            A = nn.Linear(2 * out_features, 1)
            nn.init.xavier_uniform_(W.weight, gain=1.414)
            nn.init.xavier_uniform_(A.weight, gain=1.414)
            Ws.append(W)
            As.append(A)
        self.Ws = nn.ModuleList(Ws)
        self.As = nn.ModuleList(As)
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, h):
        outs = []  # => [n_head, batch, window_size, out_features]
        for i in range(self.n_head):
            W = self.Ws[i]  # => [in_features, out_features]
            A = self.As[i]  # => [2 * out_features, 1]

            Wh = W(h)  # => [batch, window_size, out_features]
            Whi = Wh.unsqueeze(2).repeat(1, 1, Wh.size(1), 1)  # => [batch, window_size, window_size, out_features]
            Whj = Wh.unsqueeze(1).repeat(1, Wh.size(1), 1, 1)  # => [batch, window_size, window_size, out_features]
            Whi_cat_Whj = torch.cat((Whi, Whj), dim=3)  # => [batch, window_size, window_size, out_feature * 2]

            e = A(Whi_cat_Whj).squeeze()  # => [batch, window_size, window_size]
            e = self.leaky_relu(e)  # => [batch, window_size, window_size]
            a = e.softmax(dim=-1)  # => [batch, window_size, window_size]

            # np_weights = a.data.cpu().numpy()
            # for i in range(0, np_weights.shape[1]):
            #     for j in range(0, np_weights.shape[2]):
            #         print(np_weights[0, i, j].item(), end=' ')
            #     print()
            # print('ok')

            a = self.dropout(a)  # => [batch, window_size, window_size]

            aWh = torch.matmul(a, Wh)  # => [batch, window_size, out_features]
            if self.last:
                out = aWh  # => [batch, window_size, out_features]
            else:
                out = torch.relu(aWh)  # => [batch, window_size, out_features]
            outs.append(out)

        if self.last:
            outs = torch.mean(torch.stack(outs, dim=3), dim=3)  # => [batch, window_size, out_features]
        else:
            outs = torch.cat(outs, dim=2)  # => [batch, window_size, out_features * n_head]
        return torch.sigmoid(outs)


class TimeOrientedGAT(nn.Module):
    """
        Zhao H, Wang Y, Duan J, et al.
        Multivariate time-series anomaly detection via graph attention network[C]//
        2020 IEEE International Conference on Data Mining (ICDM). IEEE, 2020: 841-850.
    """

    def __init__(self, input_window_size, input_size, output_window_size,
                 gat_h_dim=64, gat_out_channels=20, gat_inner_out_channels=64, dropout=0.):
        super(TimeOrientedGAT, self).__init__()

        self.input_window_size = input_window_size
        self.input_size = input_size
        self.output_window_size = output_window_size

        self.dropout = dropout

        self.gat_inner_out_channels = gat_inner_out_channels
        self.gat_out_channels = gat_out_channels
        self.gat_h_dim = gat_h_dim

        self.gat1 = GAT(self.input_size, self.gat_inner_out_channels, n_head=self.gat_h_dim,
                        dropout_rate=self.dropout, last=False)

        self.gat2 = GAT(self.gat_inner_out_channels * self.gat_h_dim, self.gat_out_channels, n_head=self.gat_h_dim,
                        dropout_rate=self.dropout, last=False)
        self.sl = nn.Linear(self.gat_out_channels * self.gat_h_dim, self.input_size)
        self.l1 = GAR(self.input_window_size, self.output_window_size)

    def forward(self, x):
        """ x => [batch_size, input_window_size, input_size] """

        c = x  # c -> [batch_size, input_window_size, input_size]

        # GAT
        c = self.gat1(c)  # -> [batch_size, input_window_size, gat_hidden_size * gat_h_dim]
        c = torch.relu(c)  # -> [batch_size, input_window_size, gat_hidden_size * gat_h_dim]
        c = self.gat2(c)  # -> [batch_size, input_window_size, gat_out_channels * gat_h_dim]

        c = self.sl(c)  # -> [batch_size, input_window_size, input_size]
        res = self.l1(c)  # -> [batch_size, output_window_size, input_size]

        return res


class GAIN(nn.Module):
    """
        Wang Z, Liu X, Huang Y, et al. 、
        A multivariate time series graph neural network for district heat load forecasting[J].
        Energy, 2023, 278: 127911.
    """

    def __init__(self, input_window_size: int = 64, input_size: int = 30,
                 output_window_size: int = 1, output_size: int = 1,
                 gat_hidden_size: int = 64, gat_nhead: int = 128,
                 gru_hidden_size: int = 8, gru_num_layers: int = 1,
                 cnn_kernel_size: int = 3, cnn_out_channels: int = 16,
                 highway_window_size: int = 10, dropout_rate: float = 0.5):
        """
            Assure input_size === output_size. CNN -> GAT -> GRU -> Highway
        """
        super(GAIN, self).__init__()
        self.input_window_size = input_window_size
        self.input_size = input_size
        self.output_window_size = output_window_size
        self.output_size = output_size

        self.dropout_rate = dropout_rate

        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_out_channels = cnn_out_channels

        self.gat_hidden_size = gat_hidden_size
        self.gat_nhead = gat_nhead

        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers

        self.highway_window_size = min(10, highway_window_size)

        self.cnn = nn.Conv1d(self.input_size, self.cnn_out_channels,
                             self.cnn_kernel_size, padding=self.cnn_kernel_size // 2)

        self.gat = GAT(in_features=self.cnn_out_channels, out_features=self.gat_hidden_size,
                       n_head=self.gat_nhead, dropout_rate=self.dropout_rate, last=False)

        self.gru = nn.GRU(self.gat_hidden_size * self.gat_nhead + self.input_size, self.gru_hidden_size,
                          batch_first=True, num_layers=self.gru_num_layers)

        self.l1 = nn.Linear(self.gru_hidden_size, self.output_size)
        self.gar1 = GAR(self.gru_num_layers, self.output_window_size)

        self.highway = GAR(self.highway_window_size, self.output_window_size)
        self.highway_proj = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        """x -> [batch_size, input_window_size, input_size]"""

        r = x.permute(0, 2, 1)  # => [batch_size, input_size, input_window_size]
        r = self.cnn(r)  # => [batch_size, cnn_out_channels, input_window_size]
        r = r.permute(0, 2, 1)  # => [batch_size, input_window_size, cnn_out_channels]

        r = self.gat(r)  # => [batch_size, input_window_size, gat_hidden_size * gat_nhead]
        _, h = self.gru(torch.cat([r, x], dim=2))  # => [num_layers, batch_size, gru_hidden_size]
        h = h.permute(1, 0, 2)  # => [batch_size, num_layers, gru_hidden_size]

        r = self.l1(h)  # => [batch_size, num_layers, output_size]
        r = self.gar1(r)  # => [batch_size, output_window_size, output_size]

        if self.highway_window_size > 0:
            z = x[:, -self.highway_window_size:, :]  # => [batch_size, highway_window_size, input_size]
            z = self.highway(z)  # => [batch_size, output_window_size, input_size]
            z = self.highway_proj(z)  # => [batch_size, output_window_size, output_size]
            r = r + z  # => [batch_size, output_window_size, output_size]

        return r
