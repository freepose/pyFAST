#!/usr/bin/env python
# encoding: utf-8

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from torch.nn.utils.rnn import pad_sequence

from .sst_dataset import TensorSequence


@dataclass
class SampleIndex:
    ts_index: int  # index of time series source
    start: int  # start index in the time series (inclusive)
    end: int  # end index in the time series (exclusive)


def collate_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
        Iteratively collate a list of (possibly deeply nested) dictionaries.
        For each leaf key, collect values into a list.
        Does not use recursion.
    """
    if not dict_list:
        return {}

    result = {}

    # Stack of tuples: (output_dict_ref, list_of_dicts)
    stack = [(result, dict_list)]

    while stack:
        out_dict, list_of_dicts = stack.pop()

        # assume all dicts in list have the same structure
        for key in list_of_dicts[0].keys():
            values = [d[key] for d in list_of_dicts]

            # if the value itself is a dict -> go deeper
            if isinstance(values[0], dict):
                out_dict[key] = {}
                stack.append((out_dict[key], values))
            else:
                out_dict[key] = values

    return result


def time_point_alignment_or_padding(tensors: TensorSequence, batch_size: int, device: torch.device,
                                    global_time_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
        -> torch.Tensor:
    """
        Align time series in sequences based on global_time_points if provided,
        otherwise pad them to the max length in the batch.

        :param tensors: tensor sequences to be aligned or padded.
        :param batch_size: size of the batch.
        :param device: device to place the resulting tensors.
        :param global_time_points: optional tuple of (unified_time_points, inverse_indices) for alignment.
        :return: list of aligned or padded tensors.
    """

    if global_time_points is not None:
        unified_time_points, inverse_indices = global_time_points
        time_point_num = unified_time_points.shape[0]
        n_vars = tensors[0].shape[1]
        data_type = tensors[0].dtype
        align_tensor = torch.zeros((batch_size, time_point_num, n_vars), dtype=data_type, device=device)
        start, end = 0, 0
        for b, t in enumerate(tensors):
            end += t.shape[0]
            indices = inverse_indices[start:end].squeeze(1)
            align_tensor[b, indices] = t
            start = end
    else:
        align_tensor = pad_sequence(tensors, batch_first=True)

    return align_tensor


def smir_align_collate_fn(batch: List[Dict[str, Any]]) -> Tuple[TensorSequence, TensorSequence]:
    """
        Collate function for SMIrDataset.

        Batch-level time series alignment is supported, or batch-level max-length padding is used otherwise.

        :param batch: list of samples, each sample is a dict with 'inputs' and 'outputs' keys.
        :return: tuple of (input_tensors, output_tensors), each is a list of aligned or padded tensors.
    """
    batch_size = len(batch)
    batch_dict = collate_dict(batch)  # collate list of dicts to dict of lists
    ret_dict = dict({'inputs': dict(), 'outputs': dict()})

    global_input_time_points = None
    if 'input_time_points' in batch_dict:
        input_time_points_list = batch_dict.pop('input_time_points')
        global_input_time_points = torch.unique(torch.cat(input_time_points_list, dim=0), return_inverse=True)

    for k, tensors in batch_dict['inputs'].items():
        ret_dict['inputs'][k] = time_point_alignment_or_padding(tensors, batch_size,
                                                                device=tensors[0].device,
                                                                global_time_points=global_input_time_points)

    global_output_time_points = None
    if 'output_time_points' in batch_dict:
        output_time_points_list = batch_dict.pop('output_time_points')
        global_output_time_points = torch.unique(torch.cat(output_time_points_list, dim=0), return_inverse=True)

    for k, tensors in batch_dict['outputs'].items():
        ret_dict['outputs'][k] = time_point_alignment_or_padding(tensors, batch_size,
                                                                 device=tensors[0].device,
                                                                 global_time_points=global_output_time_points)

    if 'ex2' in ret_dict['inputs'] and 'ex2' in ret_dict['outputs']:
        ret_dict['inputs']['ex2'] = torch.cat([ret_dict['inputs']['ex2'], ret_dict['outputs']['ex2']], dim=1)
        del ret_dict['outputs']['ex2']

    input_list = [ret_dict['inputs'][key] for key in ret_dict['inputs'].keys()]
    output_list = [ret_dict['outputs'][key] for key in ret_dict['outputs'].keys()]

    return input_list, output_list


def compute_deltas(mask: torch.Tensor) -> torch.Tensor:
    """
        Compute time deltas from a binary mask.

        :param mask: binary mask tensor of shape (batch_size, seq_len, n_vars).
        :return: delta tensor of shape (batch_size, seq_len, n_vars).
    """
    batch_size, seq_len, n_vars = mask.shape
    deltas = torch.zeros((batch_size, seq_len, n_vars), device=mask.device, dtype=mask.dtype)

    deltas[:, 0, :] = 0  # initial delta is zero

    for t in range(1, seq_len):
        deltas[:, t, :] = (1 - mask[:, t - 1, :]) * (deltas[:, t - 1, :] + 1)

    return deltas
