#!/usr/bin/env python
# encoding: utf-8

"""
    Batch-wise Dynamic Padding (BDP) Dataset for sequence prediction.
"""
import itertools, sys, bisect
import numpy as np
import torch
import torch.utils.data as data

from typing import Literal, Tuple, List, Union
from tqdm import tqdm
from .smt_dataset import TensorSequence


class SMDDataset(data.Dataset):
    """
        Single prediction object Multiple sources sequential dataset using Dynamic-padding (SMD).

        ``SMDDataset`` pads zero values to **several** varying-length sequences in a batch.

        :param ts: list of univariate time series dataset.
        :param ts_mask: list of mask of univariate time series dataset.
        :param ex_ts: list of exogenous time series.
        :param ex_ts_mask: list of mask of exogenous time series.
        :param ex_ts2: list of another exogenous time series. Pre-known exogenous time series.
        :param input_window_size: input window size.
        :param output_window_size: output window size.
        :param horizon: the time steps between input window and output window.
        :param stride: the stride of two consecutive sliding windows.
        :param mark: the mark of the dataset, e.g., 'train', 'val', 'test'.
    """

    def __init__(self, ts: TensorSequence,
                 ts_mask: TensorSequence = None,
                 ex_ts: TensorSequence = None,
                 ex_ts_mask: TensorSequence = None,
                 ex_ts2: TensorSequence = None,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 mark: str = None):

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

        self.input_vars = ts[0].shape[1]  # fixed at 1
        self.output_vars = self.input_vars
        self.ex_vars = ex_ts[0].shape[1] if ex_ts is not None else None
        self.ex2_vars = ex_ts2[0].shape[1] if ex_ts2 is not None else None
        self.ts_num = len(ts)  # number of time series sources (a.k.a., csv file number)
        self.device = ts[0].device

        self.ratio = 1.  # the ratio of the whole dataset
        self.split_as = None  # 'train' or 'val': left part or right part of each time series
        self.mark = mark  # None denotes non-split for 'train' or 'val'

        self.ts = ts
        self.ts_mask = ts_mask
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask
        self.ex_ts2 = ex_ts2

        self.window_num_list = []  # the number of sliding windows for each time series
        self.cum_window_num_array = None  # the cumulative sum of window numbers.

        self.index_dataset(self.ts)

    def index_dataset(self, ts: TensorSequence) -> np.ndarray:
        """
            Index the dataset (list of time series).
            If the length of a time series is not zero, and <= window size, then it has only one sliding window.

            :param ts: list of time series dataset.
            :return: sample intervals of each time series. E.g., [1840, 3988, ...]
        """
        with tqdm(total=len(ts), leave=False, file=sys.stdout) as pbar:
            for i, ts in enumerate(ts):
                pbar.set_description(f'Indexing ts_{i}')
                ts_len = ts.shape[0]

                if ts_len == 0:
                    raise ValueError(f'The length of ts[{i}] is 0.')

                window_size = self.input_window_size + self.output_window_size - self.horizon + 1
                if ts_len < window_size:    # Padding is needed
                    sample_num = 1
                else:
                    sample_num = (ts_len - window_size) // self.stride + 1

                self.window_num_list.append(sample_num)

                pbar.set_postfix(window_num='{}'.format(sample_num))
                pbar.update(1)

        # Calculate the cumulative sum of window numbers.
        self.cum_window_num_array = np.cumsum(self.window_num_list)
        return self.cum_window_num_array

    def dynamic_padding(self, seq: torch.Tensor, padding: int) -> torch.Tensor:
        """
            Dynamic padding for a sequence tensor.

            :param seq: the sequence tensor to be padded with the shape (seq_len, feature_dim).
            :param padding: the number of padding elements.
            :return: the padded sequence tensor with shape (seq_len + padding, feature_dim).
        """
        padding_tensor = torch.zeros(padding, seq.shape[1], dtype=seq.dtype, device=self.device)
        padded_seq = torch.cat([seq, padding_tensor], dim=0)
        return padded_seq


    def __getitem__(self, index: int) -> Tuple[TensorSequence, TensorSequence]:
        """
            Get sequences of the dataset by index.

            :param index: the index of the dataset.
        """
        ts_index = bisect.bisect_left(self.cum_window_num_array, index)  # find the target UTS index

        # The right boundary of sample index in a TS, should
        if index == self.cum_window_num_array[ts_index]:
            ts_index += 1

        local_index = (index - self.cum_window_num_array[ts_index - 1]) if ts_index > 0 else index
        border_ts = self.ts[ts_index]

        start_x = self.stride * local_index
        end_x = start_x + self.input_window_size
        x_seq = border_ts[start_x:end_x]

        # Dynamic-padding on the right side of ``x_seq``
        if x_seq.shape[0] < self.input_window_size:
            x_seq = self.dynamic_padding(x_seq, self.input_window_size - x_seq.shape[0])

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq = border_ts[start_y:end_y]

        # Dynamic-padding on the right side of ``y_seq``
        if y_seq.shape[0] < self.output_window_size:
            y_seq = self.dynamic_padding(y_seq, self.output_window_size - y_seq.shape[0])

        input_list, output_list = [x_seq], [y_seq]

        if self.ts_mask is not None:
            local_ts_mask = self.ts_mask[ts_index]
            x_seq_mask = local_ts_mask[start_x:end_x]
            y_seq_mask = local_ts_mask[start_y:end_y]

            if x_seq_mask.shape[0] < self.input_window_size:
                x_seq_mask = self.dynamic_padding(x_seq_mask, self.input_window_size - x_seq_mask.shape[0])
            if y_seq_mask.shape[0] < self.output_window_size:
                y_seq_mask = self.dynamic_padding(y_seq_mask, self.output_window_size - y_seq_mask.shape[0])

            input_list.append(x_seq_mask)
            output_list.append(y_seq_mask)

        if self.ex_ts is not None:
            local_ex_ts = self.ex_ts[ts_index]
            ex_seq = local_ex_ts[start_x:end_x]
            if ex_seq.shape[0] < self.input_window_size:
                ex_seq = self.dynamic_padding(ex_seq, self.input_window_size - ex_seq.shape[0])
            input_list.append(ex_seq)

            if self.ex_ts_mask is not None:
                local_ex_ts_mask = self.ex_ts_mask[ts_index]
                ex_seq_mask = local_ex_ts_mask[start_x:end_x]
                if ex_seq_mask.shape[0] < self.input_window_size:
                    ex_seq_mask = self.dynamic_padding(ex_seq_mask, self.input_window_size - ex_seq_mask.shape[0])
                input_list.append(ex_seq_mask)

        if self.ex_ts2 is not None:
            local_ex_ts2 = self.ex_ts2[ts_index]
            ex2_seq_current = local_ex_ts2[start_x:end_x]
            if ex2_seq_current.shape[0] < self.input_window_size:
                ex2_seq_current = self.dynamic_padding(ex2_seq_current, self.input_window_size - ex2_seq_current.shape[0])
            ex2_seq_upcoming = local_ex_ts2[start_y:end_y]
            if ex2_seq_upcoming.shape[0] < self.output_window_size:
                ex2_seq_upcoming = self.dynamic_padding(ex2_seq_upcoming, self.output_window_size - ex2_seq_upcoming.shape[0])
            ex2_seq = torch.cat([ex2_seq_current, ex2_seq_upcoming], dim=0)

            input_list.append(ex2_seq)

        return input_list, output_list

    def __len__(self) -> int:
        return self.cum_window_num_array[-1]

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
            'sample_num': self.cum_window_num_array[-1],
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
        })

        if self.ts_mask is not None:
            params['mask'] = True

        if self.ex_ts is not None:
            params['ex_vars'] = self.ex_vars

            if self.ex_ts_mask is not None:
                params['ex_mask'] = True

        if self.ex_ts2 is not None:
            params['ex2_vars'] = self.ex2_vars

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'SMDDataset({})'.format(params_str)

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

        with tqdm(total=len(self.ts), leave=False, file=sys.stdout) as pbar:
            for i, ts in enumerate(self.ts):
                pbar.set_description(f'Splitting ts_{i}')

                ts_len = ts.shape[0]
                start, end = int(ts_len * start_ratio), int(ts_len * end_ratio)

                if not is_strict:
                    start = max(0, start - self.input_window_size - self.horizon + 1)

                border_len = end - start
                sample_num = border_len - self.input_window_size - self.output_window_size - self.horizon + 1
                sample_num = sample_num // self.stride + 1
                assert sample_num > 0, "No samples can be generated at time series {}.".format(i)

                border_ts.append(ts[start:end])

                if self.ts_mask is not None:
                    border_ts_mask.append(self.ts_mask[i][start:end])

                if self.ex_ts is not None:
                    border_ex_ts.append(self.ex_ts[i][start:end])

                    if self.ex_ts_mask is not None:
                        border_ex_ts_mask.append(self.ex_ts_mask[i][start:end])

                if self.ex_ts2 is not None:
                    border_ex_ts2.append(self.ex_ts2[i][start:end])

                self.window_num_list.append(sample_num)

                pbar.set_postfix(sample_num='{}'.format(sample_num))
                pbar.update(1)

        dataset = SMDDataset(border_ts, border_ts_mask, border_ex_ts, border_ex_ts_mask, border_ex_ts2,
                             self.input_window_size, self.output_window_size, self.horizon, self.stride, mark=mark)
        dataset.ratio = round(end_ratio - start_ratio, 15)

        return dataset
