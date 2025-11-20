#!/usr/bin/env python
# encoding: utf-8

import sys

import torch
import torch.utils.data as data

from abc import ABC
from typing import Tuple, List, Optional

from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .sst_dataset import TensorSequence
from .utils import SampleIndex

from ..data import PatchMaker


class AbstractSupervisedStrategy(ABC):
    """
        Abstract class for split strategies.
    """

    def __init__(self, mode: str):
        self.mode = mode


class ThresholdSupervisedStrategy(AbstractSupervisedStrategy):
    """
        Split strategy based on a fixed time threshold.
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__(mode='threshold')
        self.threshold: float = threshold


class PairwiseSupervisedStrategy(AbstractSupervisedStrategy):
    """
        Split strategy based on pairwise time points.
    """

    def __init__(self, return_delta: bool = True):
        super().__init__(mode='pairwise')
        self.return_delta = return_delta


class WindowSupervisedStrategy(AbstractSupervisedStrategy):
    """
        Split strategy based on fixed-size windows.
    """

    def __init__(self, input_window_size: int = 1, output_window_size: int = 1, horizon: int = 0, stride: int = 1):
        super().__init__(mode='window')
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride


class SMIrDataset(data.Dataset):
    """

        Single prediction object Multisource Irregular time series dataset (SMIrDataset).

        :param ts: time series data, a sequence of tensors, each tensor is of shape (TS_i, D).
        :param ts_mask: time series mask data, a sequence of tensors, each tensor is of shape (TS_i, D).
                        (True for observed, False for missing)
        :param timepoint_ts: time point time series, a sequence of tensors, each tensor is of shape (TS_i, 1).
        :param strategy: split strategy for the dataset.
        :param ex_ts: exogenous time series data, a sequence of tensors, each tensor is of shape (TS_i, D_ex).
        :param ex_ts_mask: exogenous time series mask data, a sequence of tensors, each tensor is of shape (TS_i, D_ex).
                           (True for observed, False for missing)
        :param ex_ts2: second exogenous time series data, a sequence of tensors, each tensor is of shape (TS_i, D_ex2).
                       It is used for preknown features, such as time features, weather features, etc.
        :param mark: a mark string for the dataset.
        :param show_progress: whether to show progress bar during indexing.
    """

    def __init__(self, ts: TensorSequence,
                 ts_mask: TensorSequence,
                 timepoint_ts: TensorSequence,
                 strategy: AbstractSupervisedStrategy = PairwiseSupervisedStrategy(),
                 ex_ts: Optional[TensorSequence] = None,
                 ex_ts_mask: Optional[TensorSequence] = None,
                 ex_ts2: Optional[TensorSequence] = None,
                 mark: str = None,
                 show_progress: bool = True):

        assert len(ts) == len(ts_mask), "The number of 'ts' and 'ts_mask' should be the same."
        assert len(ts) == len(timepoint_ts), "The number of 'ts' and 'timepoint' should be the same."

        if ex_ts is not None:
            assert len(ts) == len(ex_ts), "The number of 'ts' and 'ex_ts' should be the same."

            if ex_ts_mask is not None:
                assert len(ex_ts) == len(ex_ts_mask), "The number of 'ex_ts' and 'ex_ts_mask' should be the same."

        if ex_ts2 is not None:
            assert len(ts) == len(ex_ts2), "The number of 'ts' and 'ex_ts2' should be the same."

        self.ts = ts
        self.ts_mask = ts_mask
        self.timepoint_ts = timepoint_ts
        self.strategy = strategy
        self.ex_ts = ex_ts
        self.ex_ts_mask = ex_ts_mask
        self.ex_ts2 = ex_ts2

        self.output_vars = self.input_vars = ts[0].shape[1]
        self.timepoint_vars = self.timepoint_ts[0].shape[1]
        self.ex_vars = ex_ts[0].shape[1] if ex_ts is not None else None
        self.ex2_vars = ex_ts2[0].shape[1] if ex_ts2 is not None else None
        self.ts_num = len(ts)  # number of time series sources (a.k.a., csv file number)
        self.device = ts[0].device

        self.ratio = 1.
        self.mark = mark
        self.show_progress = show_progress

        if self.strategy.mode == 'pairwise':
            self.input_window_size = self.output_window_size = 1
        elif self.strategy.mode == 'window':
            self.input_window_size = self.strategy.input_window_size
            self.output_window_size = self.strategy.output_window_size
            self.horizon = self.strategy.horizon
            self.stride = self.strategy.stride

        self.sample_indices: List[SampleIndex] = self.index_dataset()

    def index_dataset(self):
        index_list = []
        mode = self.strategy.mode

        with (tqdm(total=self.ts_num, leave=False, file=sys.stdout, disable=not self.show_progress) as pbar):
            for sid, tp in enumerate(self.timepoint_ts):
                pbar.set_description(f'Indexing timepoint_ts[{sid}]')

                tp_len = tp.shape[0]
                if mode == 'threshold':
                    threshold = self.strategy.threshold
                    all_in = torch.all(tp <= threshold)
                    all_out = torch.all(tp > threshold)
                    if all_in:
                        raise ValueError(f"ts[{sid}] has all time points on "
                                         f"the 'inputs' side for threshold {threshold}.")
                    elif all_out:
                        raise ValueError(f"ts[{sid}] has all time points on "
                                         f"the 'outputs' side for threshold {threshold}.")
                    else:
                        index_list.append(SampleIndex(ts_index=sid, start=0, end=tp_len))

                elif mode == 'pairwise':
                    if tp_len < 2:
                        raise ValueError(f"Time series at index {sid} is too short ({tp_len}) for pairwise splitting.")

                    for j in range(tp_len - 1):
                        index_list.append(SampleIndex(ts_index=sid, start=j, end=j + 2))

                elif mode == 'window':
                    w_in = self.strategy.input_window_size
                    w_out = self.strategy.output_window_size
                    horizon = self.strategy.horizon
                    stride = self.strategy.stride

                    window_num = tp_len - (w_in + w_out + horizon - 1) + 1
                    sample_num = (window_num + stride - 1) // stride

                    if sample_num < 1:
                        raise ValueError(f"Time series at index {sid} is too short ({tp_len}) for window splitting "
                                         f"with input_window_size={w_in}, output_window_size={w_out}, "
                                         f"horizon={horizon}.")

                    for j in range(0, window_num, stride):
                        index_list.append(SampleIndex(ts_index=sid, start=j, end=j + w_in + w_out + horizon))

        return index_list

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> Tuple[TensorSequence, TensorSequence]:
        """
            Get the input and output data of the dataset by index.
        """
        sample_idx = self.sample_indices[index]
        sid, s, e = sample_idx.ts_index, sample_idx.start, sample_idx.end

        local_ts = self.ts[sid][s:e]
        local_ts_mask = self.ts_mask[sid][s:e]
        local_timepoint = self.timepoint_ts[sid][s:e]

        mode = self.strategy.mode

        inputs, outputs = [], []

        if mode == 'threshold':
            threshold = self.strategy.threshold
            x_indicator = local_timepoint.squeeze(1) <= threshold
            y_indicator = ~x_indicator

            inputs.append(local_ts[x_indicator])
            inputs.append(local_ts_mask[x_indicator])

            # maybe, add ``deltas`` here

            if self.ex_ts is not None:
                local_ex = self.ex_ts[sid][s:e]
                inputs.append(local_ex[x_indicator])
                if self.ex_ts_mask is not None:
                    local_ex_mask = self.ex_ts_mask[sid][s:e]
                    inputs.append(local_ex_mask[x_indicator])

            outputs.append(local_ts[y_indicator])
            outputs.append(local_ts_mask[y_indicator])

        elif mode == 'pairwise':
            inputs.append(local_ts[:1])
            inputs.append(local_ts_mask[:1])

            outputs.append(local_ts[1:2])
            outputs.append(local_ts_mask[1:2])

            if self.strategy.return_delta:
                delta_t = local_timepoint[1] - local_timepoint[0]
                inputs.append(delta_t.unsqueeze(0))  # shape (1, 1)

        elif mode == 'window':
            w_in = self.strategy.input_window_size
            w_out = self.strategy.output_window_size

            inputs.append(local_ts[:w_in])
            inputs.append(local_ts_mask[:w_in])

            if self.ex_ts is not None:
                local_ex = self.ex_ts[sid][s:e]
                inputs.append(local_ex[:w_in])
                if self.ex_ts_mask is not None:
                    local_ex_mask = self.ex_ts_mask[sid][s:e]
                    inputs.append(local_ex_mask[:w_in])

            start_y = w_in + self.strategy.horizon - 1
            end_y = start_y + w_out

            outputs.append(local_ts[start_y:end_y])
            outputs.append(local_ts_mask[start_y:end_y])

        return inputs, outputs

    def __str__(self):
        """
            String representation of the SMIrDataset.
        """

        params = dict()
        params['device'] = self.device
        params['ratio'] = self.ratio

        if self.mark is not None:
            params['mark'] = self.mark

        params.update(**{
            'ts_num': self.ts_num,
            'sample_num': self.__len__(),
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'timepoint_vars': self.timepoint_vars,
        })

        if isinstance(self.strategy, ThresholdSupervisedStrategy):
            params['supervised_mode'] = 'threshold'
            params['time_threshold'] = self.strategy.threshold
        elif isinstance(self.strategy, PairwiseSupervisedStrategy):
            params['supervised_mode'] = 'pairwise'
            params['input_window_size'] = 1
            params['output_window_size'] = 1
        elif isinstance(self.strategy, WindowSupervisedStrategy):
            params['supervised_mode'] = 'window'
            params['input_window_size'] = self.strategy.input_window_size
            params['output_window_size'] = self.strategy.output_window_size
            params['horizon'] = self.strategy.horizon
            params['stride'] = self.strategy.stride

        params['ts_mask'] = True

        if self.ex_ts is not None:
            params['ex_vars'] = self.ex_vars
            if self.ex_ts_mask is not None:
                params['ex_ts_mask'] = True

        if self.ex_ts2 is not None:
            params['ex2_vars'] = self.ex2_vars

        if self.show_progress:
            params['show_progress'] = True

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'SMIrDataset({})'.format(params_str)

        return params_str


def smir_collate_fn(batch: List[Tuple[TensorSequence, TensorSequence]]) -> Tuple[TensorSequence, TensorSequence]:
    """
        Collate function for SMIrDataset.
        It collates a batch of irregular time series and padded with max-length.

        :param batch: list of tuples (input, output).
        :return: tuple of input tensors and output tensors batches.
    """
    inputs, outputs = list(zip(*batch))  # transpose list of tuples to tuple of lists

    num_in_fields = len(inputs[0])
    batch_x = [pad_sequence([sample[i] for sample in inputs], batch_first=True) for i in range(num_in_fields)]

    num_out_fields = len(outputs[0])
    batch_y = [pad_sequence([sample[i] for sample in outputs], batch_first=True) for i in range(num_out_fields)]

    return batch_x, batch_y


def smir_pairwise_patch_collate_fn(batch: List[Tuple[TensorSequence, TensorSequence]]) -> Tuple[TensorSequence, TensorSequence]:
    """
        Collate function for SMIrDataset.
        :param batch: list of tuples (input, output).
        :return: tuple of input tensors and output tensors batches.
    """
    batch_x, batch_y = torch.utils.data.dataloader.default_collate(batch)


    # Reshape x, x_mask from shape (B, 1, D) to (B, D, 1),
    # and patch along the second dimension to get shape (B, 1, P, D/P)
    x, x_mask = batch_x[0].permute(0, 2, 1), batch_x[1].permute(0, 2, 1)

    patch_len = 1024

    padding_num = (patch_len - (x.shape[1] % patch_len)) % patch_len
    patch_maker = PatchMaker(x.shape[1], patch_len, patch_stride=patch_len, padding=padding_num)

    x_patched = patch_maker(x)  # shape (B, P, D/P, 1)
    x_mask_patched = patch_maker(x_mask)  # shape (B, P, D/P, 1)

    y, y_mask = batch_y[0].permute(0, 2, 1), batch_y[1].permute(0, 2, 1)
    y_patched = patch_maker(y)
    y_mask_patched = patch_maker(y_mask)

    batch_x[0] = x_patched.squeeze(1).permute(0, 2, 1)
    batch_x[1] = x_mask_patched.squeeze(1).permute(0, 2, 1)

    batch_y[0] = y_patched.squeeze(1).permute(0, 2, 1)
    batch_y[1] = y_mask_patched.squeeze(1).permute(0, 2, 1)

    return batch_x, batch_y
