#!/usr/bin/env python
# encoding: utf-8

"""
    Loading tools for time series datasets.
"""

import os, sys

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from typing import Literal, List, Tuple, Union, Dict, Any, Type

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from ... import get_device
from .. import SSTDataset, SMTDataset, SMDDataset

SSTDatasetSequence = Union[SSTDataset, Tuple[SSTDataset, ...], List[SSTDataset]]
SMTDatasetSequence = Union[SMTDataset, Tuple[SMTDataset, ...], List[SMTDataset]]
SMDDatasetSequence = Union[SMDDataset, Tuple[SMDDataset, ...], List[SMDDataset]]


def load_sst_datasets(filename: str,
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
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu') -> SSTDatasetSequence:
    """
        Load time series dataset from a **CSV** file,
        transform time series data into supervised data,
        and split the dataset into several datasets.

        The default **float type** is ``float32``, you can change it to ``float64`` if needed.
        The default **device** is ``cpu``, you can change it to ``cuda`` or ``mps`` if needed.

        :param filename: csv filename.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
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
        mask_target_tensor = ~torch.isnan(target_tensor)
        sst_args['ts_mask'] = mask_target_tensor

    if ex_variables is not None:
        ex_df = df.loc[:, ex_variables]
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_args['ex_ts'] = ex_tensor

        if mask_ex_variables:
            mask_ex_tensor = ~torch.isnan(ex_tensor)
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
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
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
            mask_target_tensor = ~torch.isnan(target_tensor)
            smt_args['ts_mask'].append(mask_target_tensor)

        if ex_variables is not None:
            ex_df = df.loc[:, ex_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            smt_args['ex_ts'].append(ex_tensor)

            if mask_ex_variables:
                mask_ex_tensor = ~torch.isnan(ex_tensor)
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


def get_smt_args_parallel(filenames: List[str],
                          variables: List[str], mask_variables: bool = False,
                          ex_variables: List[str] = None, mask_ex_variables: bool = False,
                          ex2_variables: List[str] = None,
                          float_type: np.dtype = np.float32,
                          device: torch.device = torch.device('cpu'),
                          show_progress: bool = True,
                          max_workers: int = None) -> Dict[str, Any]:
    """
        Get the arguments of **SMTDataset** dataset by reading several **CSV** files.
        Uses multi-threading to process files concurrently.

        :param filenames: list of CSV filenames.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param ex2_variables: names of the exogenous variables.
        :param float_type: the data type of the time series data, default is ``np.float32``.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
        :param show_progress: whether to show the progress bar.
                        Set to ``False`` to disable the progress bar in logging mode.
        :param max_workers: maximum number of threads, default is None (uses system default).

        :return: a dictionary of arguments for **SMTDataset**.
    """

    def process_file(filename: str) -> Dict[str, Any]:
        """Process a single CSV file and return tensors"""
        df = pd.read_csv(filename)
        result = {}

        # Process target variables
        target_df = df.loc[:, variables]
        target_array = target_df.values.astype(float_type)
        target_tensor = torch.tensor(target_array, device=device)
        result['ts'] = target_tensor

        # Process target mask
        if mask_variables:
            mask_target_array = ~np.isnan(target_array)
            mask_target_tensor = torch.tensor(mask_target_array, dtype=torch.bool, device=device)
            result['ts_mask'] = mask_target_tensor

        # Process exogenous variables
        if ex_variables is not None:
            ex_df = df.loc[:, ex_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            result['ex_ts'] = ex_tensor

            if mask_ex_variables:
                mask_ex_array = ~np.isnan(ex_array)
                mask_ex_tensor = torch.tensor(mask_ex_array, dtype=torch.bool, device=device)
                result['ex_ts_mask'] = mask_ex_tensor

        # Process second exogenous variables
        if ex2_variables is not None:
            ex_df = df.loc[:, ex2_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            result['ex_ts2'] = ex_tensor

        return result

    # Use thread_map to process files concurrently
    results = thread_map(
        process_file,
        filenames,
        max_workers=max_workers,
        desc="Loading files",
        disable=not show_progress,
        leave=False,
        file=sys.stdout
    )

    # Aggregate results
    smt_args = dict()
    smt_args['ts'] = [result['ts'] for result in results]
    smt_args['ts_mask'] = [result['ts_mask'] for result in results] if mask_variables else None
    if ex_variables is not None:
        smt_args['ex_ts'] = [result['ex_ts'] for result in results]
        if mask_ex_variables:
            smt_args['ex_ts_mask'] = [result['ex_ts_mask'] for result in results]
    smt_args['ex_ts2'] = [result['ex_ts2'] for result in results] if ex2_variables is not None else None

    return smt_args


def load_smx_datasets(filenames: List[str],
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
                      split_strategy: Literal['intra', 'inter'] = None,
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu',
                      ds_cls: Union[Type[SMTDataset], Type[SMDDataset]] = SMTDataset,
                      show_loading_progress: bool = True,
                      max_loading_workers: int = None) -> Union[SMTDatasetSequence, SMDDatasetSequence]:
    """
        Load **SMTDataset**/**SMDDataset** from several **CSV** files or directories,
        transform time series data into supervised data,
        and split the dataset into several datasets.

        :param filenames: list of CSV filenames.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion
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
        :param split_strategy: the strategy to split the dataset, can be 'intra' or 'inter', the default is 'inter'.
                                'intra' means splitting the dataset into several parts by the ratios,
                                'inter' means splitting the dataset by the filenames.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.
        :param ds_cls: the dataset class to use, default is SMTDataset.
        :param show_loading_progress: whether to show the loading progress bar.
        :param max_loading_workers: maximum number of threads for loading data, default is None (uses system default).

        :return: the (split) datasets as ``SMTDataset`` or ``SMDDataset`` objects.
    """

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

    split_strategy = split_strategy if split_strategy is not None else 'inter'
    assert split_strategy in ('intra', 'inter'), \
        f"Invalid split strategy: {split_strategy}. The split strategy should be one of ['intra', 'inter']."

    float_type = np.float32
    device = get_device(device)
    show_pregress = show_loading_progress
    max_workers = max_loading_workers

    if split_ratios is None:
        smt_args = get_smt_args_parallel(filenames, variables, mask_variables, ex_variables, mask_ex_variables,
                                         ex2_variables, float_type=float_type, device=device,
                                         show_progress=show_pregress, max_workers=max_workers)
        smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                         'horizon': horizon, 'stride': stride})
        return ds_cls(**smt_args)

    smt_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])

    if split_strategy == 'intra':
        smt_args = get_smt_args_parallel(filenames, variables, mask_variables, ex_variables, mask_ex_variables,
                                         ex2_variables, float_type=float_type,
                                         device=device, show_progress=show_pregress, max_workers=max_workers)
        smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                         'horizon': horizon, 'stride': stride})

        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            if i > 0:
                smt_args.update({'stride': output_window_size})
            split_ds = ds_cls(**smt_args).split(s, e, is_strict=False, mark='split_{}'.format(i))
            smt_datasets.append(split_ds)
    else:  # split_strategy == 'inter'
        filename_num = len(filenames)
        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            start, end = int(filename_num * round(s, 10)), int(filename_num * round(e, 10))
            split_filenames = filenames[start:end]
            smt_args = get_smt_args_parallel(split_filenames, variables, mask_variables, ex_variables,
                                             mask_ex_variables, ex2_variables, float_type=float_type,
                                             device=device, show_progress=show_pregress, max_workers=max_workers)
            smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                             'horizon': horizon, 'stride': stride})
            if i > 0:
                smt_args.update({'stride': output_window_size})
            smt_datasets.append(ds_cls(**smt_args, mark='split_{}'.format(i)))

    return smt_datasets
