#!/usr/bin/env python
# encoding: utf-8

"""
    Loading tools for time series datasets.
"""

import os
import numpy as np
import pandas as pd
import torch

from typing import Literal, List, Tuple, Union
from .time_feature import TimeAsFeature

from fast.data import SSTDataset


def load_sst_dataset(filename: str,
                     variables: List[str],
                     mask_variables: bool = False,
                     ex_variables: List[str] = None,
                     mask_ex_variables: bool = False,
                     time_variable: str = None,
                     time_feature_freq: Literal['Y', 'ME', 'W', 'd', 'B', 'h', 'min', 's'] = 'd',
                     is_time_normalized: bool = False,
                     input_window_size: int = 96,
                     output_window_size: int = 24,
                     horizon: int = 1,
                     stride: int = 1,
                     train_ratio: float = 0.8,
                     val_ratio: float = None,
                     factor: float = 1.0) -> Union[SSTDataset, Tuple[SSTDataset, ...]]:
    """
        Load time series dataset from a **CSV** file,
        transform time series data into supervised data,
        and split the dataset into training, validation, and test sets.

        The default **float type** is ``float32``, you can change it to ``float64`` if needed.
        The default **device is** ``cpu``, you can change it to ``cuda`` if needed.

        :param filename: csv filename.
        :param variables: names of the target variables.
                               If only one variable is specified, it is assumed to be the target variable.
                               If two variables are specified, the first is the start column and the last is the end column.
                                   Note: bugs happens if only **two non-adjacent variables** are need.
                               If more than two variables are specified, they are all considered as target variables.
                               The way of exogenous variables is the same.
        :param mask_variables: whether to mask the target variables. This uses for sparse time series.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse exogenous time series.
        :param time_variable: name of the time variable.
        :param time_feature_freq: frequency of the time variable.
                          The frequency should be in: ['Y', 'ME', 'W', 'd', 'B', 'h', 'min', 's'].
        :param is_time_normalized: whether to normalize the time features into [-.5, 0.5].
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param train_ratio: the ratio of training set.
        :param val_ratio: the ratio of validation set.
        :param factor: a factor to scale the **target** variables, default is 1.0.
        :return: (train_ds, val_ds, test_ds): the datasets split into training, validation, and testing sets.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'File not found: {filename}')

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    assert 0 < train_ratio <= 1, f'Invalid train_ratio: {train_ratio}. It must be in the range (0, 1].'
    if val_ratio is not None:
        assert 0 < val_ratio < 1, f'Invalid val_ratio: {val_ratio}. It must be in the range (0, 1).'
        assert 0 < train_ratio + val_ratio < 1, \
            f'Invalid test_ratio: {train_ratio} + {val_ratio}. It must be in the range (0, 1).'

    assert isinstance(factor, (int, float)), f'Invalid factor: {factor}. It must be an int or float.'

    float_type = np.float32  # the default float type is ``float32``, you can change it to ``float64`` if needed
    device = torch.device('cpu') # the default device is ``cpu``, you can change it to ``cuda`` if needed

    df = pd.read_csv(filename)

    sst_params: dict[str, Union[int, float, str, torch.Tensor]] = \
        {'input_window_size': input_window_size, 'output_window_size': output_window_size, 'horizon': horizon}

    # If only one variable is specified, it is assumed to be the target variable.
    # If two variables are specified, the first is the start column and the last is the end column.
    # Note: bugs happens if only **two non-adjacent variables** are need.
    # If more than two variables are specified, they are all considered as target variables.
    # The way of exogenous variables is the same.
    if len(variables) == 2:
        start_col, end_col = variables[0], variables[-1]
        variables = df.loc[:, start_col:end_col].columns.tolist()

    target_df = df.loc[:, variables]
    target_array = target_df.values.astype(float_type)
    target_tensor = torch.tensor(target_array, device=device)
    sst_params['ts'] = target_tensor * factor

    if mask_variables:
        mask_target_array = ~np.isnan(target_array)
        mask_target_tensor = torch.tensor(mask_target_array, dtype=torch.bool, device=device)
        sst_params['ts_mask'] = mask_target_tensor

    if ex_variables is not None:
        if len(ex_variables) == 2:
            start_col, end_col = variables[0], variables[-1]
            ex_variables = df.loc[:, start_col:end_col].columns.tolist()

        ex_df = df.loc[:, ex_variables]
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_params['ex_ts'] = ex_tensor

        if mask_ex_variables:
            mask_ex_array = ~ ~np.isnan(ex_array)
            mask_ex_tensor = torch.tensor(mask_ex_array, dtype=torch.bool, device=device)
            sst_params['ex_ts_mask'] = mask_ex_tensor

    if time_variable is not None:
        df[time_variable] = pd.to_datetime(df[time_variable])
        time_features = TimeAsFeature(freq=time_feature_freq, is_normalized=is_time_normalized)(df[time_variable].dt)
        time_feature_tensor = torch.tensor(time_features, device=device)
        sst_params["ex_ts2"] = time_feature_tensor

    if train_ratio == 1.0:
        train_ds = SSTDataset(**sst_params)
        return train_ds

    if val_ratio is None:
        train_ds = SSTDataset(**sst_params, stride=stride).split(train_ratio, 'train', 'train')
        val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size']).split(train_ratio, 'val', 'val')
        return train_ds, val_ds

    ratio1 = train_ratio + val_ratio        # split ratio of (train + val) / test
    ratio2 = train_ratio / ratio1           # split ratio of train / (train + val)
    o_stride = sst_params['output_window_size']
    train_ds = SSTDataset(**sst_params, stride=stride).split(ratio1, 'train').split(ratio2, 'train', 'train')
    val_ds = SSTDataset(**sst_params, stride=o_stride).split(ratio1, 'train').split(ratio2, 'val', 'val')
    test_ds = SSTDataset(**sst_params, stride=o_stride).split(ratio1, 'val', 'test')

    return train_ds, val_ds, test_ds
