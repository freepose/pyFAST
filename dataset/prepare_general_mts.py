#!/usr/bin/env python
# encoding: utf-8

"""

    Prepare common used time series forecasting datasets.

    (1) ETT dataset:
        Exchange rate:

    (2) Load as ``SSTDataset`` or ``SMTDataset``.

"""

import os, sys, random
from typing import Literal, List, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from fast.data import Scale, StandardScale, MinMaxScale, time_series_scaler
from fast.data import SSTDataset, SMTDataset
from dataset.time_feature import TimeAsFeature


def get_ett_columns(return_vars: Literal['univariate', 'multivariate', 'exogenous', 'all'] = 'univariate') -> List:
    """
        Get the columns of ETT dataset.

        :param return_vars: The type of variables to return. ``univariate``, ``multivariate``, ``exogenous``, and ``all``.
    """
    columns = ['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]
    elif return_vars == 'exogenous':
        return columns[1:-1]

    return columns


def load_ett_sst(ett_data_root: str, subset: Literal['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'] = 'ETTh1',
                 vars: List[str] = None,
                 ex_vars: List[str] = None, time_as_feature: TimeAsFeature = None,
                 split_ratio: float = 0.8,
                 input_window_size: int = 24, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                 scaler: Scale = None, ex_scaler: Scale = None):
    """
        Load ETT dataset as ``SSTDataset``. The train / val split type is **intra** time series.

        :param ett_data_root: The root directory of ETT dataset. For example: ``~/data/time_series/general_mts/ETT``.
        :param subset: The subset of ETT dataset, ``ETTh1``, ``ETTh2``, ``ETTm1``, and ``ETTm2``.
        :param vars: The target variable(s) of ETT dataset. Default is None, and the ``['OT']`` is chosen as target.
        :param ex_vars: The exogenous variables of ETT dataset. Default is ``None``, which means no external variables.
        :param time_as_feature: The time feature class. Default is ``None``.
        :param split_ratio: The split ratio of train and test set. Default is ``0.8``. If 1, the whole dataset will be used as train set.
        :param input_window_size: The input window size. Default is ``24``.
        :param output_window_size: The output window size. Default is ``1``.
        :param horizon: The time steps between input window and output window. Default is ``1``.
        :param stride: The time steps between two consecutive (input / output) windows. Default is ``1``.
        :param scaler: The scaler for the target variable(s). Default is ``None``, which means no scaling.
        :param ex_scaler: The scaler for the exogenous variables. Default is ``None``, which means no scaling.
        :return:
    """

    assert subset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'], "Invalid subset: {}".format(subset)
    assert 0 < split_ratio <= 1, "Invalid split ratio: {}".format(split_ratio)
    assert input_window_size > 0, "Invalid input window size: {}".format(input_window_size)
    assert output_window_size > 0, "Invalid output window size: {}".format(output_window_size)
    assert stride > 0, "Invalid stride: {}".format(stride)

    csv_file = '{}/{}.csv'.format(ett_data_root, subset)    # _20160701_20180626

    if not os.path.exists(csv_file):
        raise FileNotFoundError("File not found: {}".format(csv_file))

    df = pd.read_csv(csv_file)

    sst_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    if vars is None:
        vars = ['OT']

    target_df = df.loc[:, vars]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    target_tensor = MinMaxScale().fit_transform(target_tensor)  # Previous software were worked with this
    sst_params['ts'] = target_tensor

    if scaler is not None and type(scaler) != type(Scale()):
        scaler = scaler.fit(target_tensor)

    if time_as_feature is not None:
        df['date'] = pd.to_datetime(df['date'])
        time_dt = df['date'].dt
        time_feature_array = time_as_feature(time_dt)
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    if ex_vars is not None:
        ex_df = df.loc[:, ex_vars]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)
        sst_params['ex_ts'] = ex_tensor

        if ex_scaler is not None and type(ex_scaler) != type(Scale()):
            ex_scaler = ex_scaler.fit(ex_tensor)

    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, ex_scaler)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, ex_scaler)
