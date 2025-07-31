#!/usr/bin/env python
# encoding: utf-8

"""
    Loading tools for time series datasets.
"""

import os, sys, random, time

import numpy as np
import pandas as pd
import torch

from typing import Literal, List, Tuple, Union, Dict, Any
from pathlib import Path
from tqdm import tqdm

from ... import get_device
from .. import SSTDataset, SMTDataset
from .time_feature import TimeAsFeature


def load_sst_dataset(filename: str,
                     variables: List[str],
                     mask_variables: bool = False,
                     ex_variables: List[str] = None,
                     mask_ex_variables: bool = False,
                     ex2_variables: str = None,
                     input_window_size: int = 96,
                     output_window_size: int = 24,
                     horizon: int = 1,
                     stride: int = 1,
                     split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                     device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu') -> Union[SSTDataset, List[SSTDataset]]:
    """
        Load time series dataset from a **CSV** file,
        transform time series data into supervised data,
        and split the dataset into training, validation, and test sets.

        The default **float type** is ``float32``, you can change it to ``float64`` if needed.
        The default **device** is ``cpu``, you can change it to ``cuda`` or ``mps`` if needed.

        :param filename: csv filename.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse time series.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse exogenous time series.
        :param ex2_variables: names of the second exogenous variables. This is used for pre-known features,
                                such as time features, forecasted weather factors, etc.
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.
        :return: the (split) datasets as SSTDataset objects.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File not found: {filename}')

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

    float_type = np.float32  # the default float type is ``float32``, you can change it to ``float64`` if needed
    device = get_device(device)

    df = pd.read_csv(filename)
    sst_args = dict()

    target_df = df.loc[:, variables]
    target_array = target_df.values.astype(float_type)
    target_tensor = torch.tensor(target_array, device=device)
    sst_args['ts'] = target_tensor

    if mask_variables:
        mask_target_array = ~np.isnan(target_array)
        mask_target_tensor = torch.tensor(mask_target_array, dtype=torch.bool, device=device)
        sst_args['ts_mask'] = mask_target_tensor

    if ex_variables is not None:
        ex_df = df.loc[:, ex_variables]
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_args['ex_ts'] = ex_tensor

        if mask_ex_variables:
            mask_ex_array = ~np.isnan(ex_array)
            mask_ex_tensor = torch.tensor(mask_ex_array, dtype=torch.bool, device=device)
            sst_args['ex_ts_mask'] = mask_ex_tensor

    if ex2_variables is not None:
        ex_df = df.loc[:, ex2_variables]
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_args['ex_ts2'] = ex_tensor

    sst_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                     'horizon': horizon, 'stride': stride})

    if split_ratios is None:
        return SSTDataset(**sst_args)

    sst_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])
    for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
        if i > 0:
            sst_args.update({'stride': output_window_size})
        split_ds = SSTDataset(**sst_args).split(s, e, is_strict=False, mark='split_{}'.format(i))
        sst_datasets.append(split_ds)

    return sst_datasets


def get_smt_args(filenames: List[str],
                 variables: List[str], mask_variables: bool = False,
                 ex_variables: List[str] = None, mask_ex_variables: bool = False,
                 ex2_variables: List[str] = None,
                 float_type: np.dtype = np.float32,
                 device: torch.device = torch.device('cpu'),
                 show_progress: bool = True) -> Dict[str, Any]:
    """
        Get the arguments of **SMTDataset** dataset by reading several **CSV** files.

        :param filenames: list of CSV filenames.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse exogenous time series.
        :param ex2_variables: names of the exogenous variables.
        :param float_type: the data type of the time series data, default is ``np.float32``.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
        :param show_progress: whether to show the progress bar.
                        Set to ``False`` to disable the progress bar in logging mode.
        :return: a dictionary of arguments for **SMTDataset**.
    """

    smt_args = dict()
    smt_args['ts'] = []
    smt_args['ts_mask'] = [] if mask_variables else None
    smt_args['ex_ts'] = [] if ex_variables is not None else None
    smt_args['ex_ts_mask'] = [] if mask_ex_variables else None
    smt_args['ex_ts2'] = [] if ex2_variables is not None else None

    pbar = tqdm(total=len(filenames), leave=False, file=sys.stdout) if show_progress else None
    for filename in filenames:
        if pbar is not None:
            path_name = Path(filename)
            pbar.set_description('Loading {}'.format(path_name.parent.name + '/' + path_name.name))

        df = pd.read_csv(filename)

        target_df = df.loc[:, variables]
        target_array = target_df.values.astype(float_type)
        target_tensor = torch.tensor(target_array, device=device)
        smt_args['ts'].append(target_tensor)

        if mask_variables:
            mask_target_array = ~np.isnan(target_array)
            mask_target_tensor = torch.tensor(mask_target_array, dtype=torch.bool, device=device)
            smt_args['ts_mask'].append(mask_target_tensor)

        if ex_variables is not None:
            ex_df = df.loc[:, ex_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            smt_args['ex_ts'].append(ex_tensor)

            if mask_ex_variables:
                mask_ex_array = ~np.isnan(ex_array)
                mask_ex_tensor = torch.tensor(mask_ex_array, dtype=torch.bool, device=device)
                smt_args['ex_ts_mask'].append(mask_ex_tensor)

        if ex2_variables is not None:
            ex_df = df.loc[:, ex_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            smt_args['ex_ts2'].append(ex_tensor)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return smt_args


def load_smt_datasets(filenames: List[str],
                      variables: List[str],
                      mask_variables: bool = False,
                      ex_variables: List[str] = None,
                      mask_ex_variables: bool = False,
                      ex2_variables: List[str] = None,
                      input_window_size: int = 96,
                      output_window_size: int = 24,
                      horizon: int = 1,
                      stride: int = 1,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      split_strategy: Literal['intra', 'inter'] = 'intra',
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu') -> Union[SMTDataset, List[SMTDataset]]:
    """
        Load **SMTDataset** from several **CSV** files or directories,

        :param filenames: list of CSV filenames.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse
            exogenous time series.
        :param ex2_variables: names of the second exogenous variables.
                            This is used for pre-known features, such as time features, forecasted weather factors, etc.
        :param input_window_size: input window size of the transformed supervised data
                            A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data
                            A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70 % for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param split_strategy: the strategy to split the dataset, can be 'intra' or 'inter'.
                                'intra' means splitting the dataset into several parts by the ratios,
                                'inter' means splitting the dataset by the filenames.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.

        :return: the (split) datasets as SMTDataset objects.
    """

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

    assert split_strategy in ('intra', 'inter'), \
        f"Invalid split strategy: {split_strategy}. The split strategy should be one of ['intra', 'inter']."

    float_type = np.float32
    device = get_device(device)
    show_pregress = True

    if split_ratios is None:
        smt_args = get_smt_args(filenames, variables, mask_variables, ex_variables, mask_ex_variables, ex2_variables,
                                float_type=float_type, device=device, show_progress=show_pregress)
        smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                         'horizon': horizon, 'stride': stride})
        return SMTDataset(**smt_args)

    smt_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])

    if split_strategy == 'intra':
        smt_args = get_smt_args(filenames, variables, mask_variables, ex_variables, mask_ex_variables, ex2_variables,
                                float_type=float_type, device=device, show_progress=show_pregress)
        smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                         'horizon': horizon, 'stride': stride})

        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            if i > 0:
                smt_args.update({'stride': output_window_size})
            split_ds = SMTDataset(**smt_args).split(s, e, is_strict=False, mark='split_{}'.format(i))
            smt_datasets.append(split_ds)
    else:  # split_strategy == 'inter'
        filename_num = len(filenames)
        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            start, end = int(filename_num * s), int(filename_num * e)
            split_filenames = filenames[start:end]
            smt_args = get_smt_args(split_filenames, variables, mask_variables, ex_variables, mask_ex_variables,
                                    ex2_variables, float_type=float_type, device=device, show_progress=show_pregress)
            smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                             'horizon': horizon, 'stride': stride})
            if i > 0:
                smt_args.update({'stride': output_window_size})
            smt_datasets.append(SMTDataset(**smt_args, mark='split_{}'.format(i)))

    return smt_datasets
