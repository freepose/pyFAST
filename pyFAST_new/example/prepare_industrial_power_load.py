#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading ECPL ``STSDataset`` and ``STMDataset``.
"""

from typing import Literal, List

import numpy as np
import pandas as pd

import torch

from fast.data import Scale, StandardScale, MinMaxScale, train_test_split, time_series_scaler
from fast.data import STSDataset, STMDataset, MTMDataset


def load_industrial_power_load_sts(data_root: str,
                                   interpolate_type: Literal['li', 'mice'] = 'li',
                                   targets: List[str] = None,
                                   exogeneities: List[str] = None,
                                   ds_params: dict = None,
                                   scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load the industrial electric power load dataset as ``STSDataset``.

        The data fields are as follows:
        [time,commercial,office,public,residential,temperature,humidity], the time interval is one-hour.

    :param data_root: data directory
    :param interpolate_type: interpolation type, either 'li' or 'mice'.
    :param targets: the list of target names, one or all in ['commercial', 'office', 'public', 'residential'].
                    The default is all.
    :param exogeneities: the list of exogenous factors, maybe none, one or all in ['temperature', 'humidity'].
    :param ds_params: the common dataset parameters for train and test ``STSDataset``.
    :param scaler: the global scaler target time series and exogenous time series, default is Scale().
    :return: train and validation dataset, and the target and exogenous scalers.
    """
    pl_data_root = data_root + '/time_series/energy_electronic_power_load/2_mts_csv/'

    if targets is None:
        targets = ['commercial', 'office', 'public', 'residential']

    csv_file = pl_data_root + r'ecpl_1hour_data_{}.csv'.format(interpolate_type)
    df = pd.read_csv(csv_file)

    # time_df = df.loc[:, 'time':'time']

    target_df = df.loc[:, targets]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    target_scaler = time_series_scaler(target_tensor, scaler)

    ex_tensor, ex_scaler = None, scaler
    if exogeneities is not None:
        ex_df = df.loc[:, exogeneities]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)
        ex_scaler = time_series_scaler(ex_tensor, scaler)

    params = {'ts': target_tensor, 'ts_mask': None, 'ex_ts': ex_tensor}
    params.update(ds_params)

    train_ds = STSDataset(**params, stride=1, split='train')
    val_ds = STSDataset(**params, stride=ds_params['output_window_size'], split='val')

    return (train_ds, val_ds), (target_scaler, ex_scaler)


def load_industrial_power_load_stm(data_root: str,
                                   interpolate_type: Literal['li', 'mice'] = 'li',
                                   targets: List[str] = None,
                                   exogeneities: List[str] = None,
                                   ds_params: dict = None,
                                   scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load the industrial electric power load dataset as ``STMDataset``.

        The data fields are as follows:
        [time,commercial,office,public,residential,temperature,humidity]

        :param data_root: data directory
        :param interpolate_type: interpolation type, either 'li' or 'mice'.
        :param targets: the list of target names, one or all in ['commercial', 'office', 'public', 'residential'].
                        The default is all.
        :param exogeneities: the list of exogenous factors, maybe none, one or all in ['temperature', 'humidity'].
        :param ds_params: the common dataset parameters for train and test ``STSDataset``.
        :param scaler: the global scaler target time series and exogenous time series, default is Scale().
        :return: train and validation dataset, and the target and exogenous scalers.
    """

    pl_data_root = data_root + '/time_series/energy_electronic_power_load/2_mts_csv/'

    if targets is None:
        targets = ['commercial', 'office', 'public', 'residential']

    df = pd.read_csv(pl_data_root + r'ecpl_1hour_data_{}.csv'.format(interpolate_type))

    ts_list = []
    ex_ts_list = [] if exogeneities is not None else None

    for t in targets:
        ts_array = df[t].values.reshape(-1, 1).astype(np.float32)
        ts_tensor = torch.tensor(ts_array)
        ts_list.append(ts_tensor)

        if exogeneities is not None:
            ex_df = df.loc[:, exogeneities]
            ex_array = ex_df.values.astype(np.float32)
            ex_tensor = torch.tensor(ex_array)
            ex_ts_list.append(ex_tensor)

    target_scaler = time_series_scaler(ts_list, scaler)
    ex_scaler = time_series_scaler(ex_ts_list, scaler) if exogeneities is not None else None

    params = {'ts': ts_list, 'ex_ts': ex_ts_list}
    params.update(ds_params)

    train_ds = STMDataset(**params, stride=1, split='train')
    val_ds = STMDataset(**params, stride=ds_params['output_window_size'], split='val')

    return (train_ds, val_ds), (target_scaler, ex_scaler)
