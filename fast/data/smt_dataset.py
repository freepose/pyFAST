#!/usr/bin/env python
# encoding: utf-8
from typing import Literal, Tuple, List

import sys, bisect
import numpy as np

import torch
import torch.utils.data as data

from tqdm import tqdm


class SMTDataset(data.Dataset):
    """
        Single prediction object Multiple sources Time series dataset (SMTDataset).
        The lengths of several time series may be various.

        ``SMTDataset`` transforms **several** time series to (masked) input / output data.

        (1) Transformation from target time series to supervised (i.e., input / output) data.
        (2) Support alignment-free time series dataset.
        (3) Support input data == output data for autoencoders or generative models.
        (4) Support exogenous time series data.
        (5) Support sparse time series data: target, exogenous, and both.
        (6) Support training / validation ``split`` function.

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
    """

    def __init__(self, ts: Tuple[torch.Tensor] or List[torch.Tensor],
                 ts_mask: Tuple[torch.Tensor] or List[torch.Tensor] = None,
                 ex_ts: Tuple[torch.Tensor] or List[torch.Tensor] = None,
                 ex_ts_mask: Tuple[torch.Tensor] or List[torch.Tensor] = None,
                 ex_ts2: Tuple[torch.Tensor] or List[torch.Tensor] = None,
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

        self.ratio = 1.         # the ratio of the whole dataset
        self.split_as = None    # 'train' or 'val': left part or right part of each time series
        self.mark = mark        # None denotes non-split for 'train' or 'val'

        self.ts = ts
        self.ts_mask = ts_mask
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask
        self.ex_ts2 = ex_ts2

        self.window_num_list = []  # the number of sliding windows for each time series
        self.cum_window_num_array = None  # the cumulative sum of window numbers.

        self.index_dataset(self.ts)

    def index_dataset(self, ts) -> np.array:
        """
            Index the dataset (list of time series).
            :param ts: list of univariate time series dataset.
            :return: sample intervals of each time series. E.g., [0, 1840, 3988, ...]
        """
        with tqdm(total=len(ts), leave=False, file=sys.stdout) as pbar:
            pbar.set_description('Indexing')

            for i, ts in enumerate(ts):
                ts_len = ts.shape[0]

                sample_num = ts_len - self.input_window_size - self.output_window_size - self.horizon + 1
                sample_num = sample_num // self.stride + 1
                assert sample_num > 0, "No samples can be generated at time series {}.".format(i)

                self.window_num_list.append(sample_num)

                pbar.set_postfix(window_num='{}'.format(sample_num))
                pbar.update(1)

        # Calculate the cumulative sum of window numbers.
        self.cum_window_num_array = np.cumsum(self.window_num_list)
        return self.cum_window_num_array

    def split(self, split_ratio: float = 1.0, split_as: Literal['train', 'val'] = 'train', mark: str = None):
        """
            Split all the time series to left part (a.k.a., training set) and right part (a.k.a., validation set).
            :param split_ratio: ratio of left part (a.k.a., training set). Default is 1.0.
            :param split_as: the part of dataset, the value is 'train' or 'val', default is 'train'.
            :param mark: the mark the name of the dataset.
        """
        assert 0 < split_ratio <= 1.0, 'The split ratio must be in the range [0, 1].'
        assert split_as in ['train', 'val'], "The split type must be 'train' or 'val'."

        split_as_lookup = {'train': 0, 'val': 1}

        border_ts = []
        border_ts_mask = [] if self.ts_mask is not None else None
        border_ex_ts = [] if self.ex_ts is not None else None
        border_ex_ts_mask = [] if self.ex_ts_mask is not None else None
        border_ex_ts2 = [] if self.ex_ts2 is not None else None

        with tqdm(total=len(self.ts), leave=False, file=sys.stdout) as pbar:
            pbar.set_description('Splitting')

            for i, ts in enumerate(self.ts):
                ts_len = ts.shape[0]
                train_ts_len = int(ts_len * split_ratio)
                borders = [[0, train_ts_len], [train_ts_len - self.input_window_size - self.horizon + 1, ts_len]]
                split_border = borders[split_as_lookup[split_as]]

                start, end = split_border
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

                pbar.set_postfix(window_num='{}'.format(sample_num))
                pbar.update(1)

        dataset = SMTDataset(border_ts, border_ts_mask, border_ex_ts, border_ex_ts_mask, border_ex_ts2,
                             self.input_window_size, self.output_window_size, self.horizon, self.stride, mark=mark)

        current_ratio = split_ratio if split_as == 'train' else (1.0 - split_ratio)
        dataset.ratio = round(self.ratio * current_ratio, 15)
        dataset.split_as = split_as

        return dataset

    def __getitem__(self, index) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
            Get the input and output data of the dataset by index.
        """
        uts_index = bisect.bisect_left(self.cum_window_num_array, index)  # find the target UTS index

        # The right boundary of sample index in a UTS, should
        if index == self.cum_window_num_array[uts_index]:
            uts_index += 1

        local_index = (index - self.cum_window_num_array[uts_index - 1]) if uts_index > 0 else index
        border_ts = self.ts[uts_index]

        start_x = self.stride * local_index
        end_x = start_x + self.input_window_size
        x_seq = border_ts[start_x:end_x]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq = border_ts[start_y:end_y]

        input_list, output_list = [x_seq], [y_seq]

        if self.ts_mask is not None:
            local_ts_mask = self.ts_mask[uts_index]
            x_seq_mask = local_ts_mask[start_x:end_x]
            y_seq_mask = local_ts_mask[start_y:end_y]
            input_list.append(x_seq_mask)
            output_list.append(y_seq_mask)

        if self.ex_ts is not None:
            local_ex_ts = self.ex_ts[uts_index]
            ex_seq = local_ex_ts[start_x:end_x]
            input_list.append(ex_seq)

            if self.ex_ts_mask is not None:
                local_ex_ts_mask = self.ex_ts_mask[uts_index]
                ex_seq_mask = local_ex_ts_mask[start_x:end_x]
                input_list.append(ex_seq_mask)

        if self.ex_ts2 is not None:
            local_ex_ts2 = self.ex_ts2[uts_index]
            ex2_seq_current = local_ex_ts2[start_x:end_x]
            ex2_seq_upcoming = local_ex_ts2[start_y:end_y]
            ex2_seq = torch.cat([ex2_seq_current, ex2_seq_upcoming], dim=0)
            input_list.append(ex2_seq)

        return input_list, output_list

    def __len__(self):
        return self.cum_window_num_array[-1]

    def __str__(self) -> str:
        """
            Print the information of this class instance.
        """

        params = {
            'device': self.device,
            'ratio': self.ratio,
        }

        if self.mark is not None:
            params['mark'] = self.mark

        params.update(**{
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'sample_num': self.cum_window_num_array[-1],
            'ts_num': self.ts_num,
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
        params_str = 'SMTDataset({})'.format(params_str)

        return params_str
