#!/usr/bin/env python
# encoding: utf-8

from typing import Literal, List

import numpy as np
import pandas as pd

import torch

from fast.data import AbstractScale, scale_several_time_series
from fast.data.sst_dataset import SSTDataset


def load_vessel_trajectory_sst(csv_filename: str,
                               use_ex_vars: bool = False,
                               split_ratio: float = 0.8,
                               input_window_size: int = 10,
                               output_window_size: int = 1,
                               horizon: int = 1,
                               stride: int = 1,
                               scaler: AbstractScale = None,
                               ex_scaler: AbstractScale = None) -> tuple[tuple, tuple]:
    """

        Load vessel (15m length ship) trajectory data and prepare it for training and validation.

        The frequency is 1-second.

    """

    sst_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    vars = ['x [m]', 'y [m]']
    # vars = ['x [m]', 'y [m]', 'z [m]']
    ex_vars = ['psi[deg]', 'u [m/s]', 'v [m/s]'] if use_ex_vars else None

    df = pd.read_csv(csv_filename)

    target_df = df.loc[:, vars]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    if scaler is not None and type(scaler) != type(AbstractScale()):
        scaler = scaler.fit(target_tensor)

    if ex_vars is not None:
        ex_df = df.loc[:, ex_vars]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)
        sst_params['ex_ts'] = ex_tensor

        if ex_scaler is not None and type(ex_scaler) != type(AbstractScale()):
            ex_scaler = ex_scaler.fit(ex_tensor)

    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, stride=stride)
        return (train_ds, None), (scaler, ex_scaler)

    train_ds = SSTDataset(**sst_params, stride=stride).split(split_ratio, 'train', 'train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size']).split(split_ratio, 'val', 'val')

    return (train_ds, val_ds), (scaler, ex_scaler)
