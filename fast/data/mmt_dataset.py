#!/usr/bin/env python
# encoding: utf-8

import sys, bisect
import numpy as np

import torch
import torch.utils.data as data

from typing import Literal, List, Tuple, Union
from tqdm import tqdm
from .smt_dataset import TensorSequence

TensorSequenceSources = Union[Tuple[TensorSequence, ...], List[TensorSequence]]


class MMTDataset(data.Dataset):
    """
        Multiple prediction objects Multiple sources time series dataset (MMT).

        (1) Support multiple objects multiple sources time series with same dimensions (i.e., ``input_vars_list``).

        (2) Support input data == output data split operation, a.k.a., autoencoders and generative learning.

        (3) Support exogenous time series modeling, a.k.a., modeling with exogenous variables.

        (4) Support data missing situations, both sparse target time series and sparse exogenous time series.
        A.k.a., time series imputation, incomplete time series forecasting.

        (5) Support multiple prediction objects, and multiple exogenous data.

    """

    def __init__(self, ts: TensorSequenceSources,
                 ts_mask: TensorSequenceSources = None,
                 ex_ts: TensorSequenceSources = None,
                 ex_ts_mask: TensorSequenceSources = None,
                 input_window_size: int = 10,
                 output_window_size: int = 1,
                 horizon: int = 1,
                 stride: int = 1,
                 mark: str = None):

        if ts_mask is not None:
            assert len(ts) == len(ts_mask), "The number of ts and ts_mask should be the same."

        if ex_ts is not None:
            assert len(ts) == len(ex_ts), "The number of ts and ex_ts should be the same."

            if ex_ts_mask is not None:
                assert len(ex_ts) == len(ex_ts_mask), "The number of ex_ts and ex_ts_mask should be the same."

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride

        self.input_vars_list = [var.shape[1] for var in ts[0]]
        self.output_vars_list = self.input_vars_list
        self.ex_vars_list = [ex_var.shape[1] for ex_var in ex_ts[0]] if ex_ts is not None else None
        self.source_num = len(ts)
        self.object_num = len(ts[0])
        self.device = ts[0][0].device

        self.mark = mark  # None denotes non-split for 'train' or 'val'

        self.ts = ts
        self.ts_mask = ts_mask
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask

        self.window_num_list = []
        self.cum_window_num_array = None

        self.index_dataset(self.ts)

    def index_dataset(self, ts_sources: TensorSequenceSources) -> np.ndarray:
        """
            Index the dataset by calculating the border tensors and the number of sliding windows.
        """
        with (tqdm(total=len(ts_sources), leave=False, file=sys.stdout) as pbar):
            pbar.set_description('Indexing')

            for i, object_ts in enumerate(ts_sources):
                assert all([len(_ts) == ts_len for _ts in object_ts]), \
                    'Several time series in a source should have the same length.'

                pbar.set_description(f'Indexing source_{i}')
                ts_len = object_ts[0].shape[0]

                num = ts_len - self.input_window_size - self.output_window_size - self.horizon + 1
                num = num // self.stride + 1
                assert num > 0, 'No samples can be generated at source_{}.'.format(i)

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

        border_ts_list = self.ts[ts_index]

        start_x = self.stride * local_index
        end_x = start_x + self.input_window_size
        x_seq_list = [ts[start_x:end_x] for ts in border_ts_list]

        start_y = start_x + self.input_window_size + self.horizon - 1
        end_y = start_y + self.output_window_size
        y_seq_list = [ts[start_y:end_y] for ts in border_ts_list]

        input_list, output_list = x_seq_list, y_seq_list

        if self.ts_mask is not None:
            ts_mask_objects = self.ts_mask[ts_index]
            input_mask_list = [ts_m[start_x:end_x] for ts_m in ts_mask_objects]
            output_mask_list = [ts_m[start_y:end_y] for ts_m in ts_mask_objects]

            input_list.extend(input_mask_list)
            output_list.extend(output_mask_list)

        if self.ex_ts is not None:
            ex_ts_objects = self.ex_ts[ts_index]
            ex_input_list = [ex_ts_o[start_x:end_x] for ex_ts_o in ex_ts_objects]
            input_list.extend(ex_input_list)

            if self.ex_ts_mask is not None:
                ex_ts_mask_objects = self.ex_ts_mask[ts_index]
                ex_input_mask_list = [ex_ts_m[start_x:end_x] for ex_ts_m in ex_ts_mask_objects]
                input_list.extend(ex_input_mask_list)

        return input_list, output_list

    def __str__(self):
        """
            Print the information of this class instance.
        """
        params = dict()
        params['device'] = self.device
        # params['ratio'] = self.ratio

        if self.mark is not None:
            params['mark'] = self.mark

        params.update({
            'source_num': self.source_num,
            'object_num': self.object_num,
            'sample_num': self.cum_window_num_array[-1],
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'input_vars_list': self.input_vars_list,
            'output_vars_list': self.output_vars_list,
        })

        if self.ts_mask is not None:
            params['mask'] = True

        if self.ex_ts is not None:
            params['ex_vars_list'] = self.ex_vars_list

            if self.ex_ts_mask is not None:
                params['ex_mask'] = True

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'MMTDataset({})'.format(params_str)

        return params_str
