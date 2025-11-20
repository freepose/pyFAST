#!/usr/bin/env python
# encoding: utf-8

import numpy as np

import torch
import torch.utils.data as data

from typing import Tuple, List, Union

TensorSequence = List[torch.Tensor]
TensorOrSequence = Union[torch.Tensor, List[torch.Tensor]]


def multi_step_ahead_split(time_series: torch.Tensor, input_window_size: int = 10, output_window_size: int = 1,
                           horizon: int = 1, stride: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Transform a univariate/multivariate time series to supervised data using sliding windows.
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
    start_position_num = number_observations - input_window_size - output_window_size - horizon + 2

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


def train_test_split(tensors: List[torch.Tensor], split_ratio: float = 0.2,
                     shuffle: bool = False) -> List[torch.Tensor]:
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


class SSTDataset(data.Dataset):
    """
        Single prediction object Single source Time series dataset (SST).

        ``SSTDataset`` transforms a ** single ** (univariate or multivariate) time series to supervised (input / output) data.

        (1) Transformation from target time series to supervised (i.e., input / output) data.
        (2) Support input data == output data for autoencoders or generative models.
            ``horizon`` == 1 - ``output_window_size``.
        (3) Support exogenous time series data.
        (4) Support sparse_fusion time series data: target, exogenous, and both.
        (5) Support split for machine learning or incremental learning.

        The default device is the same as ``ts`` device.

        :param ts: univariate/multivariate time series tensor, the shape is ``[ts_len, n_vars]``.
        :param ts_mask: mask tensor of time series, the shape is ``[ts_len, n_vars]``. Support data missing situation.
        :param ex_ts: exogenous time series tensor list, each shape is ``[ts_len, ex_vars]``, support several tensors.
        :param ex_ts_mask: mask tensor of exogenous time series, the shape is ``[ts_len, ex_vars]``. Support data missing.
        :param ex_ts2: the second exogenous time series, the shape is ``[ts_len, ex2_vars]``.
                       This is designed for **pre-known** exogenous variables, e.g., time, or forecasted weather.
        :param input_window_size: the window size of samples of ** input ** tensors.
        :param output_window_size: the window size of samples of ** output ** tensors.
        :param horizon: the time step distance between x and y (maybe overlapping).
        :param stride: spacing between two consecutive (input or output) windows.
        :param mark: the mark of the dataset, default is None.
    """

    def __init__(self, ts: torch.Tensor, ts_mask: torch.Tensor = None,
                 ex_ts: torch.Tensor = None, ex_ts_mask: torch.Tensor = None,
                 ex_ts2: torch.Tensor = None,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 mark: str = None):
        assert ts.ndim == 2, "The time series must be a 2D tensor."

        if ts_mask is not None:
            assert ts.shape == ts_mask.shape, "The shape of ts and ts_mask must be the same."

        if ex_ts is not None:
            assert ts.shape[0] == ex_ts.shape[0], "The length of ts and ex_ts must be the same."

            if ex_ts_mask is not None:
                assert ex_ts.shape == ex_ts_mask.shape, "The shape of ex_ts and ex_ts_mask must be the same."

        if ex_ts2 is not None:
            assert ts.shape[0] == ex_ts2.shape[0], "The length of ts and ex_ts2 must be the same."

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.input_vars = ts.shape[1]
        self.output_vars = self.input_vars
        self.ex_vars = ex_ts.shape[1] if ex_ts is not None else None
        self.ex2_vars = ex_ts2.shape[1] if ex_ts2 is not None else None
        self.device = ts.device

        self.ratio = 1.0
        self.mark = mark    # use to mark the (split) dataset

        window_num = ts.shape[0] - self.input_window_size - self.output_window_size - self.horizon + 2
        self.sample_num = (window_num + self.stride - 1) // self.stride
        assert self.sample_num > 0, "No samples can be generated."

        self.ts = ts
        self.ts_mask = ts_mask
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask
        self.ex_ts2 = ex_ts2

    def __len__(self) -> int:
        """
            :return: the number of samples in the dataset.
        """
        return self.sample_num

    def __getitem__(self, index: int) -> Tuple[TensorSequence, TensorSequence]:
        """
            Retrieve the sample at the specified index.

            :param index: the index of the sample to be retrieved.
        """
        start_x = self.stride * index
        end_x = start_x + self.input_window_size
        x_seq = self.ts[start_x:end_x]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq = self.ts[start_y:end_y]

        input_list, output_list = [x_seq], [y_seq]

        if self.ts_mask is not None:
            x_seq_mask = self.ts_mask[start_x:end_x]
            y_seq_mask = self.ts_mask[start_y:end_y]
            input_list.append(x_seq_mask)
            output_list.append(y_seq_mask)

        if self.ex_ts is not None:
            ex_seq = self.ex_ts[start_x:end_x]
            input_list.append(ex_seq)

            if self.ex_ts_mask is not None:
                ex_seq_mask = self.ex_ts_mask[start_x:end_x]
                input_list.append(ex_seq_mask)

        if self.ex_ts2 is not None:
            # for pre-known exogenous variables, e.g., time, or pre-known forecasted weather.
            ex2_seq_current = self.ex_ts2[start_x:end_x]
            ex2_seq_upcoming = self.ex_ts2[start_y:end_y]
            ex2_seq = torch.cat([ex2_seq_current, ex2_seq_upcoming], dim=0)
            input_list.append(ex2_seq)

        return input_list, output_list

    def __str__(self) -> str:
        """
            String representation of the SSTDataset instance, including its parameters.
        """

        params = dict()
        params['device'] = self.device
        params['ratio'] = self.ratio

        if self.mark is not None:
            params['mark'] = self.mark

        params.update(**{
            'sample_num': self.sample_num,
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars
        })

        if self.ts_mask is not None:
            params['mask'] = True
            density = self.ts_mask.sum() / (self.ts_mask.shape[0] * self.ts_mask.shape[1])
            params['density'] = round(float(density), 6)

        if self.ex_ts is not None:
            params['ex_vars'] = self.ex_vars

            if self.ex_ts_mask is not None:
                params['ex_ts_mask'] = True
                ex_density = self.ex_ts_mask.sum() / (self.ex_ts_mask.shape[0] * self.ex_ts_mask.shape[1])
                params['ex_density'] = round(float(ex_density), 6)

        if self.ex_ts2 is not None:
            params['ex2_vars'] = self.ex2_vars

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'SSTDataset({})'.format(params_str)

        return params_str

    def split(self, start_ratio: float, end_ratio: float, is_strict: bool = False, mark: str = None):
        """
            Split the time series by specified boundary [start, end) for machine / incremental learning.

            :param start_ratio: the start ratio of the split boundary, must be in the range [0, 1).
            :param end_ratio: the end ratio of the split boundary, must be in the range (start_ratio, 1].
            :param is_strict: if True, the split will be strict, i.e.,
                                the start index will be exactly at the start_ratio position.
            :param mark: the mark of the split dataset for string representation, default is None.
            :return: a new SSTDataset instance with the specified split.
        """
        assert 0 <= start_ratio < end_ratio <= 1, \
            f"Invalid boundary of split ratios: {start_ratio}, {end_ratio}. They must be in the range [0, 1]."

        ts_len = self.ts.shape[0]
        start, end = int(ts_len * round(start_ratio, 10)), int(ts_len * round(end_ratio, 10))

        if not is_strict:
            start = max(0, start - self.input_window_size - self.horizon + 1)

        border_len = end - start
        window_size = self.input_window_size + self.output_window_size + self.horizon - 1
        min_border = border_len - window_size + 1
        if min_border < 0:
            raise ValueError(f"No samples can be generated in the specified range: ({start_ratio}, {end_ratio}].")

        border_ts = self.ts[start:end]
        border_ts_mask = self.ts_mask[start:end] if self.ts_mask is not None else None
        border_ex_ts = self.ex_ts[start:end] if self.ex_ts is not None else None
        border_ex_ts_mask = self.ex_ts_mask[start:end] if self.ex_ts_mask is not None else None
        border_ex_ts2 = self.ex_ts2[start:end] if self.ex_ts2 is not None else None

        dataset = SSTDataset(border_ts, border_ts_mask, border_ex_ts, border_ex_ts_mask, border_ex_ts2,
                             self.input_window_size, self.output_window_size, self.horizon, self.stride, mark=mark)
        dataset.ratio = round(end_ratio - start_ratio, 15)

        return dataset
