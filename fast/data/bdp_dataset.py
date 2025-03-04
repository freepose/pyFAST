#!/usr/bin/env python
# encoding: utf-8

"""
    Batch-wise Dynamic Padding (BDP) Dataset for sequence prediction.
"""
import sys, bisect, itertools
import numpy as np

from typing import Literal
from tqdm import tqdm

import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from .patch import PatchMaker


class BDPDataset(data.Dataset):
    """
        Single prediction Target Multiple sources (STM) sequence dataset using Batch-wise Dynamic Padding (BDP).

        ``BDPDataset`` pads zero values to **several** varying-length sequences in a batch.

        :param ts: list of univariate time series dataset.
        :param ts_mask: list of mask of univariate time series dataset.
        :param ex_ts: list of exogenous time series.
        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param horizon: the time steps between input window and output window.
        :param stride: the stride of two consecutive sliding windows.
    """

    def __init__(self, ts: tuple[torch.Tensor] or list[torch.Tensor],
                 ts_mask: tuple[torch.Tensor] or list[torch.Tensor] = None,
                 ex_ts: tuple[torch.Tensor] or list[torch.Tensor] = None,
                 ex_ts_mask: tuple[torch.Tensor] or list[torch.Tensor] = None,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1):

        if ts_mask is not None:
            assert len(ts) == len(ts_mask)

        if ex_ts is not None:
            assert len(ts) == len(ex_ts)

            if ex_ts_mask is not None:
                assert len(ts) == len(ex_ts_mask)

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.input_vars = ts[0].shape[1]  # fixed at 1
        self.output_vars = self.input_vars
        self.ex_vars = ex_ts[0].shape[1] if ex_ts is not None else None
        self.ts_num = len(ts)  # number of univariate time series (a.k.a., csv file number)

        self.device = ts[0].device

        self.ts = ts
        self.ts_mask = ts_mask
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask

    def sliding_window(self, ts: torch.Tensor, padding: int = 0) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
            Sliding window for a single time series.
            :param ts: the univariate time series.
            :param padding: the number of zero (False) padding on the right side.
            :return: [input tensor 1, input tensor 2, ...], [output tensor 1, output tensor 2, ...]
        """
        if padding > 0:
            padding_tensor = torch.zeros(padding, ts.shape[1], dtype=ts.dtype, device=self.device)
            ts = torch.cat([ts, padding_tensor], dim=0)

        start_position_num = ts.shape[0] - self.input_window_size - self.horizon - self.output_window_size + 1
        input_window_list, output_window_list = [], []
        for i in range(0, start_position_num + 1, self.stride):
            start_x = i
            end_x = i + self.input_window_size

            start_y = start_x + self.input_window_size + self.horizon - 1
            end_y = start_y + self.output_window_size

            input_window_list.append(ts[start_x:end_x])
            output_window_list.append(ts[start_y:end_y])

        return input_window_list, output_window_list

    def __getitem__(self, index) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """
            Get sequences of the dataset by index.
            :param index: the index of the dataset.
        """
        ts = self.ts[index]  # -> (ts_len, n_vars)
        ts_len, n_vars = ts.shape

        # Calculate appropriate padding based on sequence length
        if ts_len <= self.input_window_size:
            padding = self.input_window_size - ts_len
        else:
            remainder = (ts_len - self.input_window_size) % self.stride
            padding = self.stride - remainder if remainder > 0 else 0

        inputs, outputs = [], []

        ts_inputs, ts_outputs = self.sliding_window(ts, padding)
        inputs.append(ts_inputs)
        outputs.append(ts_outputs)

        if self.ts_mask is not None:
            ts_mask_inputs, ts_mask_outputs = self.sliding_window(self.ts_mask[index], padding)
            inputs.append(ts_mask_inputs)
            outputs.append(ts_mask_outputs)

        if self.ex_ts is not None:
            ex_ts_inputs, ex_ts_outputs = self.sliding_window(self.ex_ts[index], padding)
            inputs.append(ex_ts_inputs)

            if self.ex_ts_mask is not None:
                ex_ts_mask_inputs, ex_ts_mask_outputs = self.sliding_window(self.ex_ts_mask[index], padding)
                inputs.append(ex_ts_mask_inputs)

        return inputs, outputs

    def __len__(self):
        return len(self.ts)

    def __str__(self):
        """
            Print the dataset information.
        """
        params = {
            'device': self.device,
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'ts_num': self.ts_num
        }

        if self.ts_mask is not None:
            params['mask'] = True

        if self.ex_ts is not None:
            params['ex_vars'] = self.ex_vars

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'BDPDataset({})'.format(params_str)

        return params_str




def collate_fn(batch):
    """
        Collect the input and output data of the dataset by batch.
        :param batch: [[windowed sequences, ...], [mask windowed sequences, ...], [exogenous windowed sequences, ...]]
    """
    # batch = data.default_collate(batch)

    zipped_batch = list(zip(*batch))    # input list, output list

    ret_batch = []
    for li in zipped_batch:
        zipped_list = list(zip(*li))
        tensor_list = []
        for window_list in zipped_list:
            flat_list = list(itertools.chain.from_iterable(window_list))
            flat_tensor = torch.stack(flat_list)
            tensor_list.append(flat_tensor)

        ret_batch.append(tensor_list)

    return ret_batch
