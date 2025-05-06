#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading ECPL ``STSDataset`` and ``STMDataset``.
"""

from typing import Literal, List

import numpy as np
import pandas as pd

import torch

from fast.data import AbstractScale, scale_several_time_series
from fast.data import SSTDataset, SMTDataset

from dataset.time_feature import TimeAsFeature

"""
    The data fields are as follows:
        [time, commercial, office, public, residential, temperature, humidity]

    The time interval (frequency) is one-hour.
"""


def load_industrial_power_load_sst(data_root: str,
                                   interpolate_type: Literal['li', 'mice'] = 'li',
                                   vars: List[str] = None,
                                   ex_vars: List[str] = None,
                                   use_time_features: bool = False,
                                   split_ratio: float = 0.8,
                                   input_window_size: int = 8 * 24,
                                   output_window_size: int = 24,
                                   horizon: int = 1,
                                   stride: int = 1,
                                   scaler: AbstractScale = None,
                                   ex_scaler: AbstractScale = None) -> tuple[tuple, tuple]:
    """
        Load the industrial electric power load dataset as ``SSTDataset``.

        The data fields are as follows:
            [time, commercial, office, public, residential, temperature, humidity]

        The time interval (frequency) is one-hour.

        :param data_root: data directory
        :param interpolate_type: interpolation type, either 'li' or 'mice'. The missing values are interpolated by
                                linear interpolation or MICE (Multiple Imputation by Chained Equations).
        :param vars: the list of target names, one or all in ['commercial', 'office', 'public', 'residential'].
                        The default is all.
        :param ex_vars: the list of exogenous factors, maybe none, one or all in ['temperature', 'humidity'].
        :param use_time_features: whether to use time features, default is False.
        :param split_ratio: the ratio to split the target time series into train and test.
                            If ``split_ratio`` is 1.0, the whole dataset will be used as a train datasets.
        :param input_window_size: the size of the input window, default is 10.
        :param output_window_size: the size of the output window, default is 1.
        :param horizon: the size of the horizon, default is 1.
        :param stride: the stride of the dataset, default is 1.
        :param scaler: the global scaler target time series.
        :param ex_scaler: the global scaler for exogenous time series.
        :return: train and validation dataset, and the target and exogenous scalers.
    """

    sts_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'split_ratio': split_ratio,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    if vars is None:
        vars = ['commercial', 'office', 'public', 'residential']

    pl_data_root = data_root + '/time_series/energy_electronic_power_load/2_mts_csv/'
    csv_file = pl_data_root + r'ecpl_1hour_data_{}.csv'.format(interpolate_type)
    df = pd.read_csv(csv_file)

    target_df = df.loc[:, vars]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    sts_params['ts'] = target_tensor

    if scaler is not None and type(scaler) != type(AbstractScale()):
        scaler = scaler.fit(target_tensor)

    if use_time_features:
        df['time'] = pd.to_datetime(df['time'])
        taf = TimeAsFeature(freq='h', is_normalized=True)
        time_feature_array = taf(df['time'].dt).astype(np.float32)
        time_feature_tensor = torch.tensor(time_feature_array)
        sts_params['ex_ts2'] = time_feature_tensor

    if ex_vars is not None:
        ex_df = df.loc[:, ex_vars]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)
        sts_params['ex_ts'] = ex_tensor

        if ex_scaler is not None and type(ex_scaler) != type(AbstractScale()):
            ex_scaler = ex_scaler.fit(ex_tensor)

    if split_ratio == 1.0:
        train_ds = SSTDataset(**sts_params, stride=1)
        return (train_ds, None), (scaler, ex_scaler)

    train_ds = SSTDataset(**sts_params, stride=1).split(split_ratio, 'train', 'train')
    val_ds = SSTDataset(**sts_params, stride=sts_params['output_window_size']).split(split_ratio, 'val', 'val')

    return (train_ds, val_ds), (scaler, ex_scaler)


def load_industrial_power_load_smt(data_root: str,
                                   interpolate_type: Literal['li', 'mice'] = 'li',
                                   vars: List[str] = None,
                                   ex_vars: List[str] = None,
                                   use_time_features: bool = False,
                                   split_ratio: float = 0.8,
                                   input_window_size: int = 8 * 24,
                                   output_window_size: int = 24,
                                   horizon: int = 1,
                                   stride: int = 1,
                                   scaler: AbstractScale = None,
                                   ex_scaler: AbstractScale = None) -> tuple[tuple, tuple]:
    """
        Load the industrial electric power load dataset as ``SMTDataset``.

        The data fields are as follows:
            [time, commercial, office, public, residential, temperature, humidity]

        The time interval (frequency) is one-hour.

        The split type is intra time series split.

        :param data_root: data directory
        :param interpolate_type: interpolation type, either 'li' or 'mice'. The missing values are interpolated by
                                linear interpolation or MICE (Multiple Imputation by Chained Equations).
        :param vars: the list of target names, one or all in ['commercial', 'office', 'public', 'residential'].
                        The default is all.
        :param ex_vars: the list of exogenous factors, maybe none, one or all in ['temperature', 'humidity'].
        :param use_time_features: whether to use time features, default is False.
        :param split_ratio: the ratio to split the target time series into train and test.
        :param input_window_size: the size of the input window, default is 10.
        :param output_window_size: the size of the output window, default is 1.
        :param horizon: the size of the horizon, default is 1.
        :param stride: the stride of the dataset, default is 1.
        :param scaler: the global scaler target time series.
        :param ex_scaler: the global scaler for exogenous time series.
        :return: train and validation dataset, and the target and exogenous scalers.
    """
    if vars is None:
        vars = ['commercial', 'office', 'public', 'residential']

    stm_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'split_ratio': split_ratio,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
        'stride': stride,
    }

    pl_data_root = data_root + '/time_series/energy_electronic_power_load/2_mts_csv/'
    csv_file = pl_data_root + r'ecpl_1hour_data_{}.csv'.format(interpolate_type)
    df = pd.read_csv(csv_file)

    ts_list = []
    ex_ts_list = [] if ex_vars is not None else None
    ex_ts2_list = [] if use_time_features else None

    for t in vars:
        ts_array = df[t].values.reshape(-1, 1).astype(np.float32)
        ts_tensor = torch.tensor(ts_array)
        ts_list.append(ts_tensor)

        if ex_vars is not None:
            ex_df = df.loc[:, ex_vars]
            ex_array = ex_df.values.astype(np.float32)
            ex_tensor = torch.tensor(ex_array)
            ex_ts_list.append(ex_tensor)

        if use_time_features:
            df['time'] = pd.to_datetime(df['time'])
            taf = TimeAsFeature(freq='h', is_normalized=True)
            time_feature_array = taf(df['time'].dt).astype(np.float32)
            time_feature_tensor = torch.tensor(time_feature_array)
            ex_ts2_list.append(time_feature_tensor)

    if scaler is not None and type(scaler) != type(AbstractScale()):
        scaler = scale_several_time_series(ts_list, scaler)

    if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(AbstractScale()):
        ex_scaler = scale_several_time_series(ex_ts_list, ex_scaler)

    stm_params['ts'] = ts_list
    stm_params['ex_ts'] = ex_ts_list
    stm_params['ex_ts2'] = ex_ts2_list

    if split_ratio == 1.:
        train_ds = SMTDataset(**stm_params, stride=1)
        return (train_ds, None), (scaler, ex_scaler)

    train_ds = SMTDataset(**stm_params, stride=1).split(split_ratio, 'train', 'train')
    val_ds = SMTDataset(**stm_params, stride=stm_params['output_window_size']).split(split_ratio, 'val', 'val')

    return (train_ds, val_ds), (scaler, ex_scaler)
