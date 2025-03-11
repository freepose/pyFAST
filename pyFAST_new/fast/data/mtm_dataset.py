#!/usr/bin/env python
# encoding: utf-8

import sys, bisect
from typing import Literal, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data as data


class MTMDataset(data.Dataset):
    """
        Multiple prediction Targets Multiple sources (MTM) time series dataset.

        (1) Support multiple targets multiple sources time series with common dimension (i.e., ``input_vars_list``).

        (2) Support input data == output data split operation, a.k.a., autoencoders and generative learning.

        (3) Support exogenous time series modeling, a.k.a., modeling with exogenous variables.

        (4) Support data missing situations, both sparse target time series and sparse exogenous time series.
        A.k.a., time series imputation, incomplete time series forecasting.

        (5) Support multiple prediction targets, and multiple exogenous data.

        :param ts_sources: Group of time series tensors list.
                        The inter-list tensors have the same shapes, i.e., ts11.shape == ts21.shape, ``input_vars``.
                        [[ts11, ts12, ...], [ts21, ts22, ...], ...]
        :param ex_ts_sources: Group of exogenous time series tensors list.
                         The inter-list tensors have the same shapes, i.e., ``input_vars``.
                        [[ex_ts11, ex_ts12, ...], [ex_ts21, ex_ts22, ...], ...]
        :param input_window_size: The split input window size.
        :param output_window_size:The split output window size.
        :param horizon: The time step distance between input and output windows.
        :param stride: Spacing between consecutive windows.
        :param split_ratio: Split ratio of training set and test set.
        :param split: The split type, either 'train' or 'val'.
    """

    def __init__(self, ts_sources: List[List[torch.Tensor] or Tuple[torch.Tensor]],
                 ex_ts_sources: List[List[torch.Tensor or Tuple[torch.Tensor]]] = None,
                 input_window_size: int = 10,
                 output_window_size: int = 1,
                 horizon: int = 1,
                 stride: int = 1,
                 split_ratio: float = 0.8,
                 split: Literal['train', 'val'] = 'train'):

        assert 0 <= split_ratio <= 1.0 and split in ['train', 'val']
        if ex_ts_sources is not None:
            assert len(ts_sources) == len(ex_ts_sources), \
                'The length of time series sources and exogenous time series sources must be the same.'

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.split_ratio = split_ratio
        self.split = split
        self.split_position = {'train': 0, 'val': 1}

        self.ts_sources = ts_sources
        self.ex_ts_sources = ex_ts_sources

        self.input_vars_list = [var.shape[1] for var in ts_sources[0]]
        self.output_vars_list = self.input_vars_list
        self.ex_vars_list = [ex_var.shape[1] for ex_var in ex_ts_sources[0]] if ex_ts_sources is not None else None
        self.device = ts_sources[0][0].device
        self.source_num = len(ts_sources)

        self.border_ts_sources = []
        self.border_ex_ts_sources = [] if ex_ts_sources is not None else None

        self.window_num_list = []
        self.cum_window_num_array = None

        self.index_dataset(ts_sources, ex_ts_sources)

    def index_dataset(self, ts_group: List[List[torch.Tensor] or Tuple[torch.Tensor]],
                      ex_ts_group: List[List[torch.Tensor] or Tuple[torch.Tensor]] = None):
        """
            Index the dataset by calculating the border tensors and the number of sliding windows.
            :param ts_group: List of groups of multivariate time series tensors.
            :param ex_ts_group: List of groups of exogenous time series tensors.
            :return: sample intervals of each time series. E.g., [0, 1840, 3988, ...]
        """
        with (tqdm(total=len(ts_group), leave=False, file=sys.stdout) as pbar):
            pbar.set_description('Indexing')

            for i, ts_list in enumerate(ts_group):
                ts_len = ts_list[0].shape[0]
                assert all([len(ts) == ts_len for ts in ts_list]), \
                    'The time series in ts_list should have the same length.'

                train_ts_len = int(ts_len * self.split_ratio)
                borders = [[0, train_ts_len], [train_ts_len - self.input_window_size - self.horizon + 1, ts_len]]
                split_border = borders[self.split_position[self.split]]

                start, end = split_border
                border_len = end - start

                num = border_len - self.input_window_size - self.output_window_size - self.horizon + 1
                num = num // self.stride + 1
                assert num > 0, 'No samples can be generated at time series {}.'.format(i)

                self.border_ts_sources.append([ts[start:end] for ts in ts_list])

                if ex_ts_group is not None:
                    self.border_ex_ts_sources.append([ex_ts[start:end] for ex_ts in ex_ts_group[i]])

                self.window_num_list.append(num)

                pbar.set_postfix(window_num='{}'.format(num))
                pbar.update(1)

        # Calculate the cumulative sum of window numbers.
        self.cum_window_num_array = np.cumsum(self.window_num_list)
        return self.cum_window_num_array

    def __len__(self):
        return self.cum_window_num_array[-1].item()

    def __getitem__(self, index):
        """
        Get the input and output data by index.
        """
        ts_index = bisect.bisect_right(self.cum_window_num_array, index)
        local_index = index - (self.cum_window_num_array[ts_index - 1] if ts_index > 0 else 0)

        border_ts_list = self.border_ts_sources[ts_index]

        start_x = self.stride * local_index
        end_x = start_x + self.input_window_size
        x_seq_list = [ts[start_x:end_x] for ts in border_ts_list]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq_list = [ts[start_y:end_y] for ts in border_ts_list]

        input_list, output_list = x_seq_list, y_seq_list

        if self.border_ex_ts_sources is not None:
            ex_ts_list = self.border_ex_ts_sources[ts_index]
            ex_input_list = [ex_ts[start_x:end_x] for ex_ts in ex_ts_list]
            input_list.extend(ex_input_list)

        return input_list, output_list

    def __str__(self):
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
            'sample_num': self.cum_window_num_array[-1].item(),
            'input_vars_list': self.input_vars_list,
            'output_vars_list': self.output_vars_list,
            'source_num': self.source_num
        }

        if self.ex_ts_sources is not None:
            params['ex_vars_list'] = self.ex_vars_list

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'MTMDataset({})'.format(params_str)

        return params_str
