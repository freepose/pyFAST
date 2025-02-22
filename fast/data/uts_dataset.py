#!/usr/bin/env python
# encoding: utf-8

"""
    Univariate Time Series (UTS) dataset.
    (1) The transformations several univariate time series to input / output data.
    (2) The transformations several univariate time series + mask time series
        to input / output data.

    The time series length varies among different UTS.
"""
from typing import Literal

import sys, bisect
import numpy as np

import torch
import torch.utils.data as data

from tqdm import tqdm


class UTSDataset(data.Dataset):
    """
        Univariate Time Series (UTS) dataset.
        UTSDataset transforms **several** univariate time series to (masked) input / output data.

        :param uts: list of univariate time series dataset.
        :param uts_mask: list of mask of univariate time series dataset.
        :param ex_ts: list of exogenous time series.
        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param horizon: the time steps between input window and output window.
        :param stride: the stride of two consecutive sliding windows.
        :param split_ratio: ratio of training set.
        :param split: split dataset belongs to train or val.
    """

    def __init__(self, uts: tuple[torch.Tensor] or list[torch.Tensor],
                 uts_mask: tuple[torch.Tensor] or list[torch.Tensor] = None,
                 ex_ts: tuple[torch.Tensor] or list[torch.Tensor] = None,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 split_ratio: float = 0.8, split: Literal['train', 'val'] = 'train'):

        assert 0 <= split_ratio <= 1.0
        assert split in ['train', 'val']

        if uts_mask is not None:
            assert len(uts) == len(uts_mask)

        if ex_ts is not None:
            assert len(uts) == len(ex_ts)

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.split_ratio = split_ratio
        self.split = split  # split dataset belongs to train or val.
        self.split_position = {'train': 0, 'val': 1}

        self.input_vars = uts[0].shape[1]  # fixed at 1
        self.output_vars = self.input_vars
        self.ex_vars = ex_ts[0].shape[1] if ex_ts is not None else None
        self.uts_num = len(uts)  # number of univariate time series (a.k.a., csv file number)

        self.device = uts[0].device

        self.border_uts_list = []  # the border tensors of each target univariate time series.
        self.border_uts_mask_list = [] if uts_mask is not None else None  # the border tensors of masks of each TS.
        self.border_ex_ts_list = [] if ex_ts is not None else None  # border tensors of exogenous time series.

        self.window_num_list = []  # the number of sliding windows for each time series
        self.cum_window_num_array = None  # the cumulative sum of window numbers.

        self.index_dataset(uts, uts_mask, ex_ts)

    def index_dataset(self, uts: tuple[torch.Tensor] or list[torch.Tensor],
                      uts_mask: tuple[torch.Tensor] or list[torch.Tensor] = None,
                      ex_ts: tuple[torch.Tensor] or list[torch.Tensor] = None) -> np.array:
        """
            Index the dataset (list of time series).
            :param uts: list of univariate time series dataset.
            :param uts_mask: list of mask of univariate time series dataset.
            :param ex_ts: list of exogenous time series.
            :return: sample intervals of each time series. E.g., [0, 1840, 3988, ...]
        """
        with tqdm(total=len(uts), leave=False, file=sys.stdout) as pbar:
            pbar.set_description('Indexing')

            for i, ts in enumerate(uts):
                ts_len = ts.shape[0]
                train_ts_len = int(ts_len * self.split_ratio)
                borders = [[0, train_ts_len], [train_ts_len - self.input_window_size - self.horizon + 1, ts_len]]
                split_border = borders[self.split_position[self.split]]

                start, end = split_border
                border_len = end - start

                num = border_len - self.input_window_size - self.output_window_size - self.horizon + 1
                num = num // self.stride + 1
                assert num > 0, "No samples can be generated at time series {}.".format(i)

                self.border_uts_list.append(ts[start:end])

                if uts_mask is not None:
                    self.border_uts_mask_list.append(uts_mask[i][start:end])

                if ex_ts is not None:
                    self.border_ex_ts_list.append(ex_ts[i][start:end])

                self.window_num_list.append(num)

                pbar.set_postfix(window_num='{}'.format(num))
                pbar.update(1)

        # Calculate the cumulative sum of window numbers.
        self.cum_window_num_array = np.cumsum(self.window_num_list)
        return self.cum_window_num_array

    def __getitem__(self, index) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
            Get the input and output data of the dataset by index.
        """

        uts_index = bisect.bisect_left(self.cum_window_num_array, index)  # find the target UTS index

        # The right boundary of sample index in a UTS, should
        if index == self.cum_window_num_array[uts_index]:
            uts_index += 1

        local_index = (index - self.cum_window_num_array[uts_index - 1]) if uts_index > 0 else index
        border_ts = self.border_uts_list[uts_index]

        start_x = self.stride * local_index
        end_x = start_x + self.input_window_size
        x_seq = border_ts[start_x:end_x]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq = border_ts[start_y:end_y]

        input_list, output_list = [x_seq], [y_seq]

        if self.border_uts_mask_list is not None:
            local_uts_mask = self.border_uts_mask_list[uts_index]
            x_seq_mask = local_uts_mask[start_x:end_x]
            y_seq_mask = local_uts_mask[start_y:end_y]
            input_list.append(x_seq_mask)
            output_list.append(y_seq_mask)

        if self.border_ex_ts_list is not None:
            local_ex_ts = self.border_ex_ts_list[uts_index]
            ex_seq = local_ex_ts[start_x:end_x]
            input_list.append(ex_seq)

        return input_list, output_list

    def __len__(self):
        return self.cum_window_num_array[-1]

    def __str__(self) -> str:
        """
            Print the information of this class instance.
        """

        params = {
            'device': self.device,
            'split': self.split,
            'split_ratio': self.split_ratio,
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'sample_num': self.cum_window_num_array[-1],
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'uts_num': self.uts_num
        }

        if self.border_uts_mask_list is not None:
            params['mask'] = True

        if self.border_ex_ts_list is not None:
            params['ex_vars'] = self.ex_vars

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'UTSDataset({})'.format(params_str)

        return params_str
