#!/usr/bin/env python
# encoding: utf-8


import sys
import numpy as np

import torch
import torch.utils.data as data

from tqdm import tqdm
from typing import Literal, Tuple, List, Union, Optional

from .sst_dataset import TensorSequence


class SMIDataset(data.Dataset):
    """
        Single prediction object Multiple sources sequential dataset with dynamic-padding support
        for **Imputation** (SMIDataset).
        The lengths of several time series may be various.

        ``SMIDataset`` transforms **several** time series and their masks to masked input-output data,
            a.k.a., self-supervised learning. The sliding window is a special case of
            ``input_window_size == output_window_size``, and ``horizon = 1 - input_window_size`` in ``SMTDataset``.

        (1) Transformation from target time series to self-supervised (i.e., input / output, input == output) data.
            Support bidirectional sliding windows.
            Support labels. [TODO]
        (2) Support alignment-free time series dataset.
        (3) Support input data == output data for autoencoders or generative models.
            NOT support input ``ts`` != output ``ts`` for forecasting models.
        (4) Support **exogenous** time series data, such as delta_t.
        (5) Support dynamic padding for various-length time series.
        (6) Support datasets splitting for machine / incremental learning.

        :param ts: list of target time series, each with shape (seq_len, feature_dim).
        :param ts_mask_input: list of target time series masks for input, each with shape (seq_len, feature_dim).
        :param ts_mask_output: list of target time series masks for output, each with shape (seq_len, feature_dim).
        :param ex_ts: list of exogenous time series, each with shape (seq_len, feature_dim). (default: None)
        :param ex_ts_mask: list of exogenous time series masks, each with shape (seq_len, feature_dim). (default: None)
        :param window_size: the size of the sliding window. (default: 24)
        :param stride: the stride of the sliding window. (default: 24)
        :param bidirectional: whether to use bidirectional sliding windows. (default: False)
        :param dynamic_padding: whether to use dynamic padding for various-length time series. (default: False)
        :param mark: the mark of the dataset, e.g., 'train', 'val', 'test'. (default: None)
        :param tqdm_disable: whether to disable the tqdm progress bar. (default: False)
    """

    def __init__(self, ts: TensorSequence,
                 ts_mask_input: TensorSequence,
                 ts_mask_output: TensorSequence,
                 ex_ts: TensorSequence = None,
                 ex_ts_mask: TensorSequence = None,
                 window_size: int = 24,
                 stride: int = 24,
                 bidirectional: bool = False,
                 dynamic_padding: bool = False,
                 mark: str = None,
                 tqdm_disable: bool = False):

        if ex_ts is not None:
            if len(ex_ts) != len(ts):
                raise ValueError(f'The number of exogenous time series ({len(ex_ts)}) must '
                                 f'be equal to the number of target time series ({len(ts)}).')

            if ex_ts_mask is not None:
                if len(ex_ts_mask) != len(ex_ts):
                    raise ValueError(f'The number of exogenous time series masks ({len(ex_ts_mask)}) '
                                     f'must be equal to the number of exogenous time series ({len(ex_ts)}).')

        assert window_size > 0, f"'window_size' ({window_size}) should be greater than 1."
        assert stride > 0, f"'stride' ({stride}) should be greater than 0."

        self.window_size = window_size
        self.stride = stride
        self.bidirectional = bidirectional
        self.dynamic_padding = dynamic_padding

        self.ratio = 1.  # the ratio of the whole dataset
        self.mark = mark  # use to mark the (split) dataset
        self.tqdm_disable = tqdm_disable

        self.output_vars = self.input_vars = ts[0].shape[1]
        self.input_window_size = self.output_window_size = window_size  # for reusable of forecasting models
        self.ex_vars = ex_ts[0].shape[1] if ex_ts is not None else None
        self.ts_num = len(ts)  # number of time series sources (a.k.a., csv file number)
        self.device = ts[0].device

        self.ts = ts
        self.ts_mask_input = ts_mask_input
        self.ts_mask_output = ts_mask_output
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask

        self.cum_sample_num_array = None  # the cumulative sum of window numbers.

        self.input_density = None  # the density of the total input time series.
        self.output_density = None  # the density of the total output time series.
        self.ex_density = None  # the density of the total exogenous time series.

        self.index_dataset()

    @staticmethod
    def compute_padding_num(seq_len: int, window_size: int, stride: int) -> int:
        """
            Compute the number of padding elements for a sequence.

            :param seq_len: the length of the sequence.
            :param window_size: the size of the sliding window.
            :param stride: the stride of the sliding window.

            :return: the number of padding elements.
        """
        base = max(0, window_size - seq_len)
        align = (stride - ((max(seq_len, window_size) - window_size) % stride)) % stride

        return base + align

    def pad_sequence(self, seq: torch.Tensor, padding_num: int) -> torch.Tensor:
        """
            Dynamic padding for a sequence (tensor).

            :param seq: the sequence tensor to be padded with the shape (seq_len, feature_dim).
            :param padding_num: the number of padding elements.

            :return: the padded sequence tensor with shape (seq_len + padding_num, feature_dim).
        """
        padding_tensor = torch.zeros(padding_num, seq.shape[1], dtype=seq.dtype, device=self.device)
        padded_seq = torch.cat([seq, padding_tensor], dim=0)

        return padded_seq

    def index_dataset(self):
        """
            Index the dataset (list of time series).
            :return: sample intervals of each time series. E.g., [1840, 3988, ...]
        """
        input_numerator = input_denominator = 0
        output_numerator = output_denominator = 0
        ex_numerator = ex_denominator = 0

        with tqdm(total=len(self.ts), leave=False, file=sys.stdout, disable=self.tqdm_disable) as pbar:
            sample_num_list = []
            for i, local_ts in enumerate(self.ts):
                pbar.set_description(f'Indexing ts[{i}]')
                local_ts_len = local_ts.shape[0]

                if self.dynamic_padding:
                    padding_num = self.compute_padding_num(local_ts_len, self.window_size, self.stride)
                    if padding_num > 0:
                        local_ts = self.pad_sequence(local_ts, padding_num)

                        local_ts_len = local_ts.shape[0]    # update the length of the time series
                        self.ts[i] = local_ts               # update the time series

                        local_ts_mask_input = self.ts_mask_input[i]
                        local_ts_mask_input = self.pad_sequence(local_ts_mask_input, padding_num)
                        self.ts_mask_input[i] = local_ts_mask_input

                        local_ts_mask_output = self.ts_mask_output[i]
                        local_ts_mask_output = self.pad_sequence(local_ts_mask_output, padding_num)
                        self.ts_mask_output[i] = local_ts_mask_output

                        if self.ex_ts is not None:
                            local_ex_ts = self.ex_ts[i]
                            local_ex_ts = self.pad_sequence(local_ex_ts, padding_num)
                            self.ex_ts[i] = local_ex_ts

                            if self.ex_ts_mask is not None:
                                local_ex_ts_mask = self.ex_ts_mask[i]
                                local_ex_ts_mask = self.pad_sequence(local_ex_ts_mask, padding_num)
                                self.ex_ts_mask[i] = local_ex_ts_mask

                window_num = local_ts_len - self.window_size + 1
                sample_num = (window_num + self.stride - 1) // self.stride
                assert sample_num > 0, (f"No samples can be generated at time series {i}. "
                                        f"'dynamic_padding' = {self.dynamic_padding}")

                sample_num_list.append(sample_num)

                local_ts_mask_input = self.ts_mask_input[i]
                input_numerator += local_ts_mask_input.sum()
                input_denominator += (local_ts_mask_input.shape[0] * local_ts_mask_input.shape[1])

                local_ts_mask_output = self.ts_mask_output[i]
                output_numerator += local_ts_mask_output.sum()
                output_denominator += (local_ts_mask_output.shape[0] * local_ts_mask_output.shape[1])

                if self.ex_ts_mask is not None:
                    local_ex_ts_mask = self.ex_ts_mask[i]
                    ex_numerator += local_ex_ts_mask.sum()
                    ex_denominator += (local_ex_ts_mask.shape[0] * local_ex_ts_mask.shape[1])

                pbar.set_postfix(sample_num='{}'.format(sample_num))
                pbar.update(1)

        self.input_density = input_numerator / input_denominator
        self.output_density = output_numerator / output_denominator
        self.ex_density = (ex_numerator / ex_denominator) if self.ex_ts_mask is not None else None

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
        local_ts_mask_input = self.ts_mask_input[ts_index]
        local_ts_mask_output = self.ts_mask_output[ts_index]

        start = self.stride * local_index
        end = start + self.window_size

        x_seq = local_ts[start:end]
        x_seq_mask_input = local_ts_mask_input[start:end]
        x_seq_mask_output = local_ts_mask_output[start:end]

        input_list, output_list = [x_seq, x_seq_mask_input], [x_seq, x_seq_mask_output]

        if self.ex_ts is not None:
            local_ex_ts = self.ex_ts[ts_index]
            ex_seq = local_ex_ts[start:end]
            input_list.append(ex_seq)

            if self.ex_ts_mask is not None:
                local_ex_ts_mask = self.ex_ts_mask[ts_index]
                ex_seq_mask = local_ex_ts_mask[start:end]
                input_list.append(ex_seq_mask)

        return input_list, output_list

    def __str__(self):
        """

        """
        params = dict()
        params['device'] = self.device
        params['ratio'] = self.ratio

        if self.mark is not None:
            params['mark'] = self.mark

        params.update(**{
            'ts_num': self.ts_num,
            'sample_num': self.cum_sample_num_array[-1],
            'window_size': self.window_size,
            'stride': self.stride,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'input_density': round(float(self.input_density), 6),
            'output_density': round(float(self.output_density), 6)
        })

        if self.ex_ts is not None:
            params['ex_vars'] = self.ex_vars

            if self.ex_ts_mask is not None:
                params['ex_ts_mask'] = True
                if self.ex_density is not None:
                    params['ex_density'] = round(float(self.ex_density), 6)

        if self.bidirectional:
            params['bidirectional'] = self.bidirectional

        if self.dynamic_padding:
            params['dynamic_padding'] = self.dynamic_padding

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'SMTDataset({})'.format(params_str)

        return params_str

    def split(self, start_ratio: float = 0.0, end_ratio: float = 1.0, mark: str = None):
        """

        """
        assert 0 <= start_ratio < end_ratio <= 1, \
            f"Invalid boundary of split ratios: {start_ratio}, {end_ratio}. They must be in the range [0, 1]."

        border_ts = []
        border_ts_mask_input = []
        border_ts_mask_output = []
        border_ex_ts = [] if self.ex_ts is not None else None
        border_ex_ts_mask = [] if self.ex_ts_mask is not None else None

        with tqdm(total=len(self.ts), leave=False, file=sys.stdout, disable=self.tqdm_disable) as pbar:
            for i, local_ts in enumerate(self.ts):
                pbar.set_description(f'Splitting ts[{i}]')

                local_ts_len = local_ts.shape[0]
                start, end = int(local_ts_len * round(start_ratio, 10)), int(local_ts_len * round(end_ratio, 10))

                window_num = local_ts_len - self.window_size + 1
                sample_num = (window_num + self.stride - 1) // self.stride
                if sample_num < 1:
                    raise ValueError(f"No samples can be generated at time series {i} "
                                     f"in the specified range: ({start_ratio}, {end_ratio}].")

                local_ts_mask_input = self.ts_mask_input[i]
                local_ts_mask_output = self.ts_mask_output[i]

                border_ts.append(local_ts[start:end])
                border_ts_mask_input.append(local_ts_mask_input[start:end])
                border_ts_mask_output.append(local_ts_mask_output[start:end])

                if self.ex_ts is not None:
                    local_ex_ts = self.ex_ts[i]
                    border_ex_ts.append(local_ex_ts[start:end])
                    if self.ex_ts_mask is not None:
                        local_ex_ts_mask = self.ex_ts_mask[i]
                        border_ex_ts_mask.append(local_ex_ts_mask[start:end])

                pbar.set_postfix({"ts": i, "sample_num": sample_num})
                pbar.update(1)

        border_dataset = SMIDataset(border_ts, border_ts_mask_input, border_ts_mask_output,
                                    ex_ts=border_ex_ts, ex_ts_mask=border_ex_ts_mask,
                                    window_size=self.window_size, stride=self.stride,
                                    bidirectional=self.bidirectional,
                                    dynamic_padding=self.dynamic_padding,
                                    mark=mark, tqdm_disable=self.tqdm_disable)
        border_dataset.ratio = round(end_ratio - start_ratio, 15)

        return border_dataset
