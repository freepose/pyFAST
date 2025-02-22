#!/usr/bin/env python
# encoding: utf-8

"""
    Single Time Series (STS) dataset.
    (1) The transformations on single univariate/multivariate time series to input / output data.
    (2) The transformations on single univariate/multivariate time series + mask time series
        to input / output data.
    (3) The transformations on single univariate/multivariate time series + exogenous time series
        to input / output data.
    (4) The transformations on single univariate/multivariate time series + mask time series +
        exogenous time series to input / output data.
"""

from typing import Literal

import numpy as np

import torch
import torch.utils.data as data


def multi_step_ahead_split(time_series: torch.tensor, input_window_size: int = 10, output_window_size: int = 1,
                           horizon: int = 1, stride: int = 1) -> tuple:
    """
        Transform a univariate/multivariate time series to supervised data using slicing windows.
        Support both multi-horizon and multistep ahead data split.
        :param time_series:         time series feature (a 2d torch array), shape is ``[ts_len, n_vars]``.
        :param input_window_size:   number of y samples to give model
        :param output_window_size:  number of future y samples to predict.
        :param horizon:             the time step distance between x and y, default is 1,
                                    maybe overlapping if horizon < 1.
        :param stride:              spacing between two consecutive windows (default is 1).
        :return X, Y:               arrays for model learning.
                                    X shape is ``[sample_num, input_window_size, n_vars]``,
                                    Y shape is ``[sample_num, output_window_size, n_vars]``.
    """
    number_observations, number_time_series = time_series.shape
    start_position_num = number_observations - input_window_size - output_window_size - horizon + 1

    inputs_list, output_list = [], []
    for i in range(0, start_position_num + 1, stride):
        start_x, start_y = i, i + input_window_size + horizon - 1

        x = time_series[start_x:start_x + input_window_size]
        y = time_series[start_y:start_y + output_window_size]

        inputs_list.append(x)
        output_list.append(y)

    inputs = torch.stack(inputs_list, dim=0)
    outputs = torch.stack(output_list, dim=0)

    return inputs, outputs


def train_test_split(tensors: list or tuple, split_ratio: float = 0.2, shuffle: bool = False) -> list or tuple:
    """
        Split supervised dataset to train / test sets.
        :param tensors: numpy arrays or torch tensors to be split.
        :param split_ratio: split ratio of training set and test set of all tensors.
        :param shuffle: shuffle flag. If ``True``, shuffle the tensors before splitting.
        :return:
    """
    assert all(len(tensors[0]) == len(tensor) for tensor in tensors), "Size mismatch between tensors"

    if shuffle:
        indices = np.arange(len(tensors[0]))
        np.random.shuffle(indices)
        tensors = [tensor[indices] for tensor in tensors]

    split_pos = round(len(tensors[0]) * split_ratio)

    left_array_list, right_array_list = [], []
    for array in tensors:
        left_part = array[:split_pos]  # left part is training set
        right_part = array[split_pos:]  # right part is test set

        left_array_list.append(left_part)
        right_array_list.append(right_part)

    return left_array_list + right_array_list


class STSDataset(data.Dataset):
    """
        Single Time Series (STS) dataset.
        STSDataset transforms a **single** univariate/multivariate time series to (masked) input / output data.
        The default device is the same as ``ts`` device.
        :param ts: univariate/multivariate time series tensor, the shape is ``[ts_len, n_vars]``.
        :param ts_mask: mask tensor of time series, the shape is ``[ts_len, n_vars]``. Support data missing situation.
        :param ex_ts: exogenous time series tensor list, each shape is ``[ts_len, ...]``, support several tensors.
        :param ex_ts_mask: mask tensor of exogenous time series, the shape is ``[ts_len, ...]``. Support data missing.
        :param input_window_size: the window size of samples of **input** tensors.
        :param output_window_size: the window size of samples of **output** tensors.
        :param horizon: the time step distance between x and y (maybe overlapping).
        :param stride: spacing between two consecutive (input or output) windows.
        :param split_ratio: split ratio of training set and test set.
        :param split: the split type of dataset, the value is 'train' or 'val'.
    """

    def __init__(self, ts: torch.Tensor, ts_mask: torch.Tensor = None,
                 ex_ts: torch.Tensor = None, ex_ts_mask: torch.Tensor = None,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 split_ratio: float = 0.8, split: Literal['train', 'val'] = 'train'):
        assert 0 <= split_ratio <= 1.0 and split in ['train', 'val']

        if ts_mask is not None:
            assert ts.shape == ts_mask.shape, "The shape of ts and ts_mask must be the same."

        if ex_ts is not None:
            assert ts.shape[0] == ex_ts.shape[0], "The length of ts and ex_ts must be the same."

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.split_ratio = split_ratio
        self.split = split
        self.split_position = {'train': 0, 'val': 1}

        self.input_vars = ts.shape[1]
        self.output_vars = self.input_vars
        self.ex_vars = ex_ts.shape[1] if ex_ts is not None else None
        self.device = ts.device

        ts_len = ts.shape[0]
        train_ts_len = int(ts_len * self.split_ratio)
        borders = [[0, train_ts_len], [train_ts_len - input_window_size - horizon + 1, ts_len]]
        split_border = borders[self.split_position[self.split]]

        start, end = split_border
        self.border_ts = ts[start:end]
        self.border_ts_mask = ts_mask[start:end] if ts_mask is not None else None
        self.border_ex_ts = ex_ts[start:end] if ex_ts is not None else None
        self.border_ex_ts_mask = ex_ts_mask[start:end] if ex_ts_mask is not None else None

        border_len = end - start
        self.sample_num = (border_len - input_window_size - output_window_size - horizon + 1) // stride + 1
        assert self.sample_num > 0, "No samples can be generated."

    def __len__(self) -> int:
        return self.sample_num

    def __getitem__(self, index) -> tuple[list, list]:
        start_x = self.stride * index
        end_x = start_x + self.input_window_size
        x_seq = self.border_ts[start_x:end_x]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq = self.border_ts[start_y:end_y]

        input_list, output_list = [x_seq], [y_seq]

        if self.border_ts_mask is not None:
            x_seq_mask = self.border_ts_mask[start_x:end_x]
            y_seq_mask = self.border_ts_mask[start_y:end_y]
            input_list.append(x_seq_mask)
            output_list.append(y_seq_mask)

        if self.border_ex_ts is not None:
            ex_seq = self.border_ex_ts[start_x:end_x]
            input_list.append(ex_seq)

            if self.border_ts_mask is not None:
                ex_seq_mask = self.border_ex_ts_mask[start_x:end_x]
                input_list.append(ex_seq_mask)

        return input_list, output_list

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
            'sample_num': self.sample_num,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars
        }

        if self.border_ts_mask is not None:
            params['mask'] = True

        if self.border_ex_ts is not None:
            params['ex_vars'] = self.ex_vars

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'STSDataset({})'.format(params_str)

        return params_str
