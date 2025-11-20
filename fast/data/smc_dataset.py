#!/usr/bin/env python
# encoding: utf-8

"""
    Single-object Multi-source Coordinate-matrix Time Series Dataset (SMCDataset).

    This is designed for ex_mask and irregularly sampled time series data, such as PhysioNet.

    There are two situations of time points for the ex_mask/irregularly sampled time series data:
        (1) All time series share common time points, but some values are missing (e.g., PhysioNet).
        (2) All time series have their own time points, and the time points are different across time series.
            (Before developing in this version)
"""

import sys
import numpy as np
import torch
import torch.utils.data as data

from tqdm import tqdm
from typing import Tuple, Dict, Union, List, Optional

from .sst_dataset import TensorSequence


class SMCDataset(data.Dataset):
    """
        Single-object Multi-source Coordinate-matrix Time Series Dataset (SMCDataset).

        This is designed for irregularly sampled time series data, such as PhysioNet, MIMIC-III, etc.

        ``SMCDataset`` assumes all time series share common (global) time points,
                     and transforms **several** time series to masked input / output data.

        :param coo: shape (N, 3), the coordinates of the time series data.
                    The three columns are (ts_index (int), time_point_index (int), variable_index (int)).
        :param values: shape (N, 1), the values of the time series data.
        :param input_window_size: int, the size of the input window.
        :param output_window_size: int, the size of the output window.
        :param horizon: int, the horizon of the output window.
        :param stride: int, the stride of the sliding window.
        :param global_ts_ids: shape (num_global_ts,), the global time series IDs.
                                If None, use the unique time series IDs in the coo.
        :param global_time_point_ids: shape (num_global_time_points,), the global time point IDs.
                                If None, use the unique time point IDs in the coo.
        :param global_variable_ids: shape (num_global_variables,), the global variable IDs.
                                If None, use the unique variable IDs in the coo.
        :param mark: str, the mark of the dataset.
        :param tqdm_disable: bool, whether to disable the tqdm progress bar.
    """

    def __init__(self, coo: torch.Tensor,
                 values: torch.Tensor,
                 input_window_size: int = 10, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 global_ts_ids: Optional[Union[List[int], torch.Tensor]] = None,
                 global_time_point_ids: Optional[Union[List[int], torch.Tensor]] = None,
                 global_variable_ids: Optional[Union[List[int], torch.Tensor]] = None,
                 mark: Optional[str] = None,
                 tqdm_disable: bool = False):

        assert coo.shape[0] == values.shape[0], 'The number of coordinates must match the number of values.'
        assert coo.ndim == 2 and coo.shape[1] == 3, 'The shape of coordinates must be (N, 3).'

        self.coo = coo  # shape is (N, 3) -> (ts_index (int), time_point_index (int), variable_index (int))
        self.values = values  # shape is (N,) -> (value (float),)

        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.horizon = horizon
        self.stride = stride
        self.mark = mark
        self.tqdm_disable = tqdm_disable
        self.device = self.values.device  # dataset device

        # Index grouping of coo matrix
        self.u_ts_ids, self.ts_boundaries, self.u_time_point_ids, self.u_variable_ids = self.grouping(self.coo)

        self.global_ts_ids = self.u_ts_ids if global_ts_ids is None else global_ts_ids
        self.global_time_point_ids = self.u_time_point_ids if global_time_point_ids is None else global_time_point_ids
        self.global_variable_ids = self.u_variable_ids if global_variable_ids is None else global_variable_ids

        self.global_ts_num = len(self.global_ts_ids)
        self.global_time_point_num = len(self.global_time_point_ids)
        self.global_variable_num = len(self.global_variable_ids)
        self.output_vars = self.input_vars = len(self.global_variable_ids)

        # Indexing the dataset
        self.cum_window_num_array = self.index_dataset(self.coo, self.ts_boundaries)

        self.ratio = 1.0
        self.density = self.coo.shape[0] / (self.global_ts_num * self.global_time_point_num)

    def grouping(self, coo: torch.Tensor) -> Tuple:
        """
            Group the coordinates by time series IDs, and get the boundaries of each time series in the coo matrix.

            Assumes the coo is already sorted by (ts, time). If not, please sort it before calling this function.

            :param coo: shape (N, 3), the coordinates of the time series data.
            :return:
                u_ts_ids: unique time series IDs, shape (num_ts,)
                ts_boundaries: dict of time series ID to (start_index, end_index) in the coo matrix
                u_time_point_ids: unique time point IDs, shape (num_time_points,)
                u_variable_ids: unique variable IDs, shape (num_variables,)

        """
        ts_ids_sorted = coo[:, 0]  # Given the coo is already sorted by (ts, time).

        unique_ts_ids, counts = torch.unique_consecutive(ts_ids_sorted, return_counts=True)
        starts = torch.cat([torch.tensor([0], device=self.device, dtype=torch.long), counts.cumsum(0)[:-1]])
        ends = counts.cumsum(0)

        ts_boundaries = {int(ts_id): (int(s), int(e)) for ts_id, s, e in zip(unique_ts_ids, starts, ends)}

        unique_time_point_ids = torch.unique(coo[:, 1])
        unique_variable_ids = torch.unique(coo[:, 2])

        return unique_ts_ids, ts_boundaries, unique_time_point_ids, unique_variable_ids

    def index_dataset(self, coo: torch.Tensor, ts_boundaries: Dict[int, Tuple[int, int]]) -> np.ndarray:
        """
            Index the dataset by calculating the cumulative number of windows for each time series.
            Assumes all time series share common (global) time points.

            :param coo: shape (N, 3), the coordinates of the time series data.
            :param ts_boundaries: dict of time series ID to (start_index, end_index) in the coo matrix

            :return: cum_window_num_array: cumulative sum of window numbers, shape (num_ts,)
        """
        with tqdm(total=len(ts_boundaries), leave=False, file=sys.stdout, disable=self.tqdm_disable) as pbar:
            window_num_list = []
            for ts_id, (start, end) in ts_boundaries.items():
                pbar.set_description(f'Indexing {ts_id}')

                # Assumes time points are sorted and continuous
                # local_time_points = self.coo[start:end, 1]
                # ts_len = int(local_time_points[-1] - local_time_points[0])

                ts_len = self.global_time_point_num

                window_size = ts_len - self.input_window_size - self.output_window_size - self.horizon + 2
                sample_num = (window_size + self.stride - 1) // self.stride
                assert sample_num > 0, f"No samples can be generated at time series {ts_id}."

                window_num_list.append(sample_num)

                pbar.set_postfix({"ts_id": ts_id, "windows": sample_num})
                pbar.update(1)

        cum_window_num_array = np.cumsum(window_num_list)  # cumulative sum of window numbers
        return cum_window_num_array

    def __len__(self) -> int:
        return int(self.cum_window_num_array[-1])

    def __getitem__(self, index: int) -> Tuple[TensorSequence, TensorSequence]:
        """
            Get the input and output data of the dataset by index.

            The empty x_seq and y_seq will be filled with zeros.

            NOTE if ``x_seq`` or ``y_seq`` are all zeros, which means no observation in the input or output window.
        """

        ts_index = int(np.searchsorted(self.cum_window_num_array, index, side='right'))

        coo_start, coo_end = self.ts_boundaries[int(self.u_ts_ids[ts_index])]
        local_coo = self.coo[coo_start:coo_end]
        local_values = self.values[coo_start:coo_end]
        local_index = (index - self.cum_window_num_array[ts_index - 1]) if ts_index > 0 else index

        x_start_time_index = self.stride * local_index
        x_end_time_index = x_start_time_index + self.input_window_size

        x_time_points = (local_coo[:, 1] >= x_start_time_index) & (local_coo[:, 1] < x_end_time_index)
        x_local_coo = local_coo[x_time_points]

        x_seq = torch.zeros(self.input_window_size, self.global_variable_num, device=local_coo.device)
        x_seq_mask = torch.zeros(self.input_window_size, self.global_variable_num, dtype=torch.bool, device=local_values.device)
        x_time_indices = x_local_coo[:, 1] - x_start_time_index  # Reset the time index to start from zero
        x_seq[x_time_indices, x_local_coo[:, 2]] = local_values[x_time_points].squeeze(1)
        x_seq_mask[x_time_indices, x_local_coo[:, 2]] = True

        y_start_time_index = x_end_time_index + self.horizon
        y_end_time_index = y_start_time_index + self.output_window_size

        y_time_points = (local_coo[:, 1] >= y_start_time_index) & (local_coo[:, 1] < y_end_time_index)
        y_local_coo = local_coo[y_time_points]

        y_seq = torch.zeros(self.output_window_size, self.global_variable_num, device=local_values.device)
        y_seq_mask = torch.zeros(self.output_window_size, self.global_variable_num, dtype=torch.bool, device=local_values.device)
        y_time_indices = y_local_coo[:, 1] - y_start_time_index  # Reset the time index to start from zero
        y_seq[y_time_indices, y_local_coo[:, 2]] = local_values[y_time_points].squeeze(1)
        y_seq_mask[y_time_indices, y_local_coo[:, 2]] = True

        return [x_seq, x_seq_mask], [y_seq, y_seq_mask]

    def __str__(self):
        """
            String representation of the SMCDataset.
        """
        params = dict()
        params['device'] = self.device
        params['ratio'] = self.ratio

        if self.mark is not None:
            params['mark'] = self.mark

        global_groups = (self.global_ts_num, self.global_time_point_num, self.global_variable_num)
        local_groups = (self.u_ts_ids.shape[0], self.u_time_point_ids.shape[0], self.u_variable_ids.shape[0])

        params.update(**{
            'coo': tuple(self.coo.shape),
            'values': tuple(self.values.shape),
            'global/local': '{}/{}'.format(global_groups, local_groups),
            'sample_num': self.cum_window_num_array[-1],
            'input_window_size': self.input_window_size,
            'output_window_size': self.output_window_size,
            'horizon': self.horizon,
            'stride': self.stride,
            'input_vars': self.input_vars,
            'output_vars': self.output_vars,
            'density': round(float(self.density), 6),
        })

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'SMCDataset({})'.format(params_str)

        return params_str

    def split(self, start_ratio: float = 0.0, end_ratio: float = 1.0, is_strict: bool = False, mark: str = None):
        """
        """
        assert 0 <= start_ratio < end_ratio <= 1, \
            f"Invalid boundary of split ratios: {start_ratio}, {end_ratio}. They must be in the range [0, 1]."

        border_coo_list = []
        border_values_list = []

        # All time series share common (global) time points
        ts_len = self.global_time_point_num
        start, end = int(ts_len * start_ratio), int(ts_len * end_ratio)

        with tqdm(total=len(self.ts_boundaries), leave=False, file=sys.stdout, disable=self.tqdm_disable) as pbar:
            for ts_id, (coo_start, coo_end) in self.ts_boundaries.items():
                pbar.set_description(f'Splitting {ts_id}')

                local_coo = self.coo[coo_start:coo_end]
                local_values = self.values[coo_start:coo_end]

                if not is_strict:
                    start = max(0, start - self.input_window_size - self.horizon + 1)

                # Check if there are enough time points to form a sample
                border_len = end - start
                window_size = border_len - self.input_window_size - self.output_window_size - self.horizon + 1
                sample_num = window_size // self.stride + 1
                assert sample_num > 0, "No samples can be generated at time series {}.".format(ts_id)

                # Get the border coo and values
                border_indicator = (local_coo[:, 1] >= start) & (local_coo[:, 1] < end)
                border_coo = local_coo[border_indicator]
                border_values = local_values[border_indicator]

                border_coo_list.append(border_coo)
                border_values_list.append(border_values)

                pbar.set_postfix({"ts_id": ts_id, "sample_num": sample_num})
                pbar.update(1)

        border_coo_tensor = torch.cat(border_coo_list, dim=0)
        border_values_tensor = torch.cat(border_values_list, dim=0)

        split_time_point_ids = self.global_time_point_ids[start:end]
        split_dataset = SMCDataset(border_coo_tensor, border_values_tensor,
                                   input_window_size=self.input_window_size,
                                   output_window_size=self.output_window_size,
                                   horizon=self.horizon,
                                   stride=self.stride,
                                   global_ts_ids=self.global_ts_ids,
                                   global_time_point_ids=split_time_point_ids,
                                   global_variable_ids=self.global_variable_ids,
                                   mark=mark,
                                   tqdm_disable=self.tqdm_disable)

        split_dataset.ratio = round(end_ratio - start_ratio, 15)

        return split_dataset
