#!/usr/bin/env python
# encoding: utf-8

import sys
import numpy as np

import torch
import torch.utils.data as data

from tqdm import tqdm
from typing import Literal, Tuple, List, Union

from .sst_dataset import TensorSequence


class SMTDataset(data.Dataset):
    """
        Single prediction object Multiple sources Time series dataset (SMTDataset).
        The lengths of several time series may be various.

        ``SMTDataset`` transforms **several** time series to (masked) input / output data.

        (1) Transformation from target time series to supervised (i.e., input / output) data.
        (2) Support alignment-free time series dataset.
        (3) Support input data == output data for autoencoders or generative models.
        (4) Support **exogenous** time series data.
        (5) Support **sparse_fusion** time series data: target, exogenous, and both.
        (6) Support datasets splitting for machine / incremental learning.

        :param ts: list of univariate/multivariate time series dataset.
        :param ts_mask: list of mask of univariate/multivariate time series dataset.
        :param ex_ts: list of exogenous time series.
        :param ex_ts_mask: list of mask of exogenous time series.
        :param ex_ts2: list of second exogenous time series. For pre-known future exogenous data, e.g., time.
        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param horizon: the time steps between input window and output window.
        :param stride: the stride of two consecutive sliding windows.
        :param mark: the mark of the dataset, default is None.
        :param tqdm_disable: whether to disable the tqdm progress bar, default is False.
    """

    def __init__(self, ts: TensorSequence,
                 ts_mask: TensorSequence = None,
                 ex_ts: TensorSequence = None,
                 ex_ts_mask: TensorSequence = None,
                 ex_ts2: TensorSequence = None,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 mark: str = None,
                 tqdm_disable: bool = False):

        if ts_mask is not None:
            assert len(ts) == len(ts_mask), "The number of ts and ts_mask should be the same."

        if ex_ts is not None:
            assert len(ts) == len(ex_ts), "The number of ts and ex_ts should be the same."

            if ex_ts_mask is not None:
                assert len(ex_ts) == len(ex_ts_mask), "The number of ex_ts and ex_ts_mask should be the same."

        if ex_ts2 is not None:
            assert len(ts) == len(ex_ts2), "The number of ts and second ex_ts2 should be the same."

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.ratio = 1.  # the ratio of the whole dataset
        self.mark = mark  # use to mark the (split) dataset
        self.tqdm_disable = tqdm_disable

        self.output_vars = self.input_vars = ts[0].shape[1]
        self.ex_vars = ex_ts[0].shape[1] if ex_ts is not None else None
        self.ex2_vars = ex_ts2[0].shape[1] if ex_ts2 is not None else None
        self.ts_num = len(ts)  # number of time series sources (a.k.a., csv file number)
        self.device = ts[0].device

        self.ts = ts
        self.ts_mask = ts_mask
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask
        self.ex_ts2 = ex_ts2

        self.cum_sample_num_array = None  # the cumulative sum of window numbers.

        self.density = None  # the density of the target time series
        self.ex_density = None  # the density of the exogenous time series

        self.index_dataset()

    def index_dataset(self) -> np.ndarray:
        """
            Index the dataset (list of time series).
            :param ts: list of time series dataset.
            :return: sample intervals of each time series. E.g., [1840, 3988, ...]
        """
        numerator = denominator = 0
        ex_numerator = ex_denominator = 0

        with tqdm(total=len(self.ts), leave=False, file=sys.stdout, disable=self.tqdm_disable) as pbar:
            sample_num_list = []
            for i, local_ts in enumerate(self.ts):
                pbar.set_description(f'Indexing ts[{i}]')
                local_ts_len = local_ts.shape[0]
                # assert ts > 0, f"The length of ts[{i}] should be larger than 0."

                window_num = local_ts_len - self.input_window_size - self.output_window_size - self.horizon + 2
                sample_num = (window_num + self.stride - 1) // self.stride
                assert sample_num > 0, f"No samples can be generated at time series {i}."

                sample_num_list.append(sample_num)

                if self.ts_mask is not None:
                    local_ts_mask = self.ts_mask[i]
                    numerator += local_ts_mask.sum()
                    denominator += (local_ts_mask.shape[0] * local_ts_mask.shape[1])

                if self.ex_ts_mask is not None:
                    local_ex_ts_mask = self.ex_ts_mask[i]
                    ex_numerator += local_ex_ts_mask.sum()
                    ex_denominator += (local_ex_ts_mask.shape[0] * local_ex_ts_mask.shape[1])

                pbar.set_postfix(sample_num='{}'.format(sample_num))
                pbar.update(1)

        self.density = numerator * 1.0 / denominator if denominator > 0 else None
        self.ex_density = ex_numerator * 1.0 / ex_denominator if ex_denominator > 0 else None

        # Calculate the cumulative sum of window numbers.
        self.cum_sample_num_array = np.cumsum(sample_num_list)

        return self.cum_sample_num_array

    def __len__(self) -> int:
        return self.cum_sample_num_array[-1]

    def __getitem__(self, index: int) -> Tuple[TensorSequence, TensorSequence]:
        """
            Get the input and output data of the dataset by index.
        """
        ts_index = int(np.searchsorted(self.cum_sample_num_array, index, side='right'))

        local_index = (index - self.cum_sample_num_array[ts_index - 1]) if ts_index > 0 else index
        local_ts = self.ts[ts_index]

        start_x = self.stride * local_index
        end_x = start_x + self.input_window_size
        x_seq = local_ts[start_x:end_x]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq = local_ts[start_y:end_y]

        input_list, output_list = [x_seq], [y_seq]

        if self.ts_mask is not None:
            local_ts_mask = self.ts_mask[ts_index]
            x_seq_mask = local_ts_mask[start_x:end_x]
            y_seq_mask = local_ts_mask[start_y:end_y]
            input_list.append(x_seq_mask)
            output_list.append(y_seq_mask)

        if self.ex_ts is not None:
            local_ex_ts = self.ex_ts[ts_index]
            ex_seq = local_ex_ts[start_x:end_x]
            input_list.append(ex_seq)

            if self.ex_ts_mask is not None:
                local_ex_ts_mask = self.ex_ts_mask[ts_index]
                ex_seq_mask = local_ex_ts_mask[start_x:end_x]
                input_list.append(ex_seq_mask)

        if self.ex_ts2 is not None:
            local_ex_ts2 = self.ex_ts2[ts_index]
            ex2_seq_current = local_ex_ts2[start_x:end_x]
            ex2_seq_upcoming = local_ex_ts2[start_y:end_y]
            ex2_seq = torch.cat([ex2_seq_current, ex2_seq_upcoming], dim=0)
            input_list.append(ex2_seq)

        return input_list, output_list

    def __str__(self) -> str:
        """
            String representation of the SMTDataset.
        """

        params = dict()
        params['device'] = self.device
        params['ratio'] = self.ratio

        if self.mark is not None:
            params['mark'] = self.mark

        params.update(**{
            'ts_num': self.ts_num,
            'sample_num': self.cum_sample_num_array[-1],
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
        })

        if self.ts_mask is not None:
            params['mask'] = True
            if self.density is not None:
                params['density'] = round(float(self.density), 6)

        if self.ex_ts is not None:
            params['ex_vars'] = self.ex_vars

            if self.ex_ts_mask is not None:
                params['ex_ts_mask'] = True
                if self.ex_density is not None:
                    params['ex_density'] = round(float(self.ex_density), 6)

        if self.ex_ts2 is not None:
            params['ex2_vars'] = self.ex2_vars

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'SMTDataset({})'.format(params_str)

        return params_str

    def split(self, start_ratio: float = 0.0, end_ratio: float = 1.0, is_strict: bool = False, mark: str = None):
        """
            Split every time series by specified boundary [start, end) for machine / incremental learning.
            :param start_ratio: the start ratio of the split boundary, must be in the range [0, 1).
            :param end_ratio: the end ratio of the split boundary, must be in the range (start_ratio, 1].
            :param is_strict: if True, the split will be strict, i.e.,
                                the start index will be exactly at the start_ratio position.
            :param mark: the mark of the split dataset for string representation, default is None.
            :return: a new SMTDataset instance with the specified split.
        """
        assert 0 <= start_ratio < end_ratio <= 1, \
            f"Invalid boundary of split ratios: {start_ratio}, {end_ratio}. They must be in the range [0, 1]."

        border_ts = []
        border_ts_mask = [] if self.ts_mask is not None else None
        border_ex_ts = [] if self.ex_ts is not None else None
        border_ex_ts_mask = [] if self.ex_ts_mask is not None else None
        border_ex_ts2 = [] if self.ex_ts2 is not None else None

        with tqdm(total=len(self.ts), leave=False, file=sys.stdout, disable=self.tqdm_disable) as pbar:
            for i, local_ts in enumerate(self.ts):
                pbar.set_description(f'Splitting ts[{i}]')

                ts_len = local_ts.shape[0]
                start, end = int(ts_len * round(start_ratio, 10)), int(ts_len * round(end_ratio, 10))

                if not is_strict:
                    start = max(0, start - self.input_window_size - self.horizon + 1)

                window_num = ts_len - self.input_window_size - self.output_window_size - self.horizon + 2
                sample_num = (window_num + self.stride - 1) // self.stride
                if sample_num < 1:
                    raise ValueError(f"No samples can be generated at time series {i} "
                                     f"in the specified range: ({start_ratio}, {end_ratio}].")

                border_ts.append(local_ts[start:end])

                if self.ts_mask is not None:
                    border_ts_mask.append(self.ts_mask[i][start:end])

                if self.ex_ts is not None:
                    border_ex_ts.append(self.ex_ts[i][start:end])

                    if self.ex_ts_mask is not None:
                        border_ex_ts_mask.append(self.ex_ts_mask[i][start:end])

                if self.ex_ts2 is not None:
                    border_ex_ts2.append(self.ex_ts2[i][start:end])

                pbar.set_postfix({"ts": i, "sample_num": sample_num})
                pbar.update(1)

        dataset = SMTDataset(border_ts, border_ts_mask, border_ex_ts, border_ex_ts_mask, border_ex_ts2,
                             self.input_window_size, self.output_window_size, self.horizon, self.stride, mark=mark)
        dataset.ratio = round(end_ratio - start_ratio, 15)

        return dataset
