#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn
from torch import Tensor

import torchquantum as tq

from .deep_time import RidgeRegressor


class QINRLayer(nn.Module):
    def __init__(self, n_wires: int, n_blocks: int, time_size: int):
        super().__init__()
        self.n_wires = n_wires
        self.n_blocks = n_blocks
        self.norm = nn.BatchNorm1d(time_size)

        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires, n_blocks):
                super().__init__()
                self.n_wires = n_wires
                self.n_blocks = n_blocks
                self.measurez = tq.MeasureAll(tq.PauliZ)

                encoder = []
                index = 0
                for i in range(n_wires):
                    encoder.append({'input_idx': [index], 'func': 'rx', 'wires': [i]})
                    index += 1
                    encoder.append({'input_idx': [index], 'func': 'ry', 'wires': [i]})
                    index += 1
                    encoder.append({'input_idx': [index], 'func': 'rz', 'wires': [i]})
                    index += 1
                    encoder.append({'input_idx': [index], 'func': 'rx', 'wires': [i]})
                    index += 1

                self.encoder = tq.GeneralEncoder(encoder)

                self.rz1_layers = tq.QuantumModuleList()
                self.ry1_layers = tq.QuantumModuleList()
                self.rz2_layers = tq.QuantumModuleList()
                self.rz3_layers = tq.QuantumModuleList()
                self.cnot_layers = tq.QuantumModuleList()

                for _ in range(n_blocks):
                    self.rz1_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RZ,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=True,
                        )
                    )
                    self.ry1_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RY,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=True,
                        )
                    )
                    self.rz2_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RZ,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=False,
                        )
                    )
                    self.rz3_layers.append(
                        tq.Op1QAllLayer(
                            op=tq.RZ,
                            n_wires=n_wires,
                            has_params=True,
                            trainable=False,
                        )
                    )
                    self.cnot_layers.append(
                        tq.Op2QAllLayer(
                            op=tq.CNOT,
                            n_wires=n_wires,
                            has_params=False,
                            trainable=False,
                            circular=True,
                        )
                    )

            def forward(self, x):
                x = x.squeeze()
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)

                for k in range(self.n_blocks):
                    self.rz1_layers[k](qdev)
                    self.ry1_layers[k](qdev)
                    self.rz2_layers[k](qdev)
                    self.cnot_layers[k](qdev)

                    if k != self.n_blocks - 1:
                        self.encoder(qdev, x)

                out = [self.measurez(qdev).unsqueeze(0)]

                for i in range(self.n_wires):
                    qdev.h(wires=i)
                out.append(self.measurez(qdev).unsqueeze(0))

                for i in range(self.n_wires):
                    qdev.h(wires=i)
                    qdev.sx(wires=i)
                out.append(self.measurez(qdev).unsqueeze(0))

                self.rz3_layers[-1](qdev)
                out.append(self.measurez(qdev).unsqueeze(0))

                out = torch.cat(out, dim=-1)
                return out

        self.QNN = QLayer(self.n_wires, self.n_blocks)
        self.QLinear = nn.Linear(self.n_wires * 4, self.n_wires * 4)
        self.CLinear = nn.Linear(self.n_wires * 4, self.n_wires * 4)
        self.linear = nn.Linear(self.n_wires * 8, self.n_wires * 4)

    def forward(self, x):

        x1 = self._qlayer(x)
        x2 = self._clayer(x)
        out = self.linear(torch.cat([x1, x2], dim=-1))
        return out

    def _qlayer(self, x: Tensor) -> Tensor:
        return self.QNN(self.norm(self.QLinear(x)))

    def _clayer(self, x: Tensor) -> Tensor:
        return torch.relu(self.CLinear(x))


class QINR(nn.Module):
    def __init__(self, in_feats: int, time_size: int, layers: int, n_wires: int, n_blocks: int):
        super().__init__()
        clayers = [QINRLayer(n_wires, n_blocks, time_size) for _ in range(layers)]
        self.layers = nn.Sequential(*clayers)
        self.features = nn.Linear(in_feats, n_wires * 4)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return self.layers(x)


class EchoStateNetworkv2(nn.Module):

    def __init__(self, input_size, reservoir_size, spectral_radius=0.9, leaking_rate=0.3, input_scaling=1.0):
        super().__init__()

        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.input_size = input_size

        self.ridgeregressor = RidgeRegressor()

        self.W_res = torch.rand(reservoir_size, reservoir_size).cuda() - 0.5
        self.W_res *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W_res)))
        self.W_in = torch.rand(input_size, reservoir_size).cuda() - 0.5
        self.W_in *= input_scaling

    def forward(self, input_data, target_data):
        reservoir_states = self.run_reservoir(input_data)
        w, b = self.ridgeregressor(reservoir_states, target_data)
        return w, b

    def predict(self, input_data):
        reservoir_states = self.run_reservoir(input_data)
        return reservoir_states

    def run_reservoir(self, input_data):
        windows_size = 8
        overlap_size = 2
        out_size = windows_size - 2 * overlap_size

        padding1 = torch.zeros((input_data.shape[0], overlap_size, input_data.shape[-1])).cuda()
        x = torch.cat([padding1, input_data], dim=1)
        x = torch.cat([x, padding1], dim=1)
        x = x.unfold(1, windows_size, out_size).transpose(2, 3)
        reservoir_states = torch.zeros((x.shape[0], x.shape[1], windows_size - overlap_size, self.reservoir_size))

        for t in range(0, windows_size - overlap_size):
            reservoir_states[:, :, t, :] = (1 - self.leaking_rate) * reservoir_states[:, :, t - 1, :] + \
                                           self.leaking_rate * torch.sin(
                torch.matmul(reservoir_states[:, :, t - 1, :], self.W_res) +
                torch.matmul(x[:, :, t, :], self.W_in),
            ).squeeze()

        reservoir_states = reservoir_states[:, :, overlap_size:, :].reshape(input_data.shape[0], input_data.shape[1],
                                                                            self.reservoir_size)
        return reservoir_states


class QuantumTime(nn.Module):
    def __init__(self, in_feats: int, time_size: int, layers: int, layer_size: int, n_blocks: int):
        super().__init__()

        self.qinr = QINR(in_feats=in_feats + 1, time_size=time_size, layers=layers, n_wires=int(layer_size / 4),
                         n_blocks=n_blocks)
        self.adaptive_weights = EchoStateNetworkv2(layer_size, 256, 0.9, 0.3, 0.5)

    def forward(self, x: Tensor, x_time: Tensor, y_time: Tensor) -> Tensor:
        tgt_horizon_len = y_time.shape[1]
        batch_size, lookback_len, _ = x.shape
        coords = self.get_coords(lookback_len, tgt_horizon_len).to(x.device)
        if y_time.shape[-1] != 0:
            time = torch.cat([x_time, y_time], dim=1)
            # coords = repeat(coords, '1 t 1 -> b t 1', b=time.shape[0])
            coords = coords.repeat(time.shape[0], 1, 1)
            coords = torch.cat([coords, time], dim=-1)

            time_reprs = self.qinr(coords)
        else:

            # time_reprs = repeat(self.qinr(coords), '1 t d -> b t d', b=batch_size)
            time_reprs = self.inr(coords).repeat(batch_size, 1, 1)

        lookback_reprs = time_reprs[:, :-tgt_horizon_len]
        horizon_reprs = time_reprs[:, -tgt_horizon_len:]
        w, b = self.adaptive_weights(lookback_reprs, x)
        h = self.adaptive_weights.predict(horizon_reprs)
        preds = self.forecast(h, w, b)

        return preds

    def forecast(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        return torch.einsum('... d o, ... t d -> ... t o', [w, inp]) + b

    def get_coords(self, lookback_len: int, horizon_len: int) -> Tensor:
        coords = torch.linspace(0, 1, lookback_len + horizon_len)
        coords = coords.unsqueeze(1).unsqueeze(0)  # rearrange(coords, 't -> 1 t 1')

        return coords
