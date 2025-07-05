#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading kaggle-turkey-wpf as ``SSTDataset``.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Literal, List, Union

from fast.data import SSTDataset
from experiment.time_feature import TimeAsFeature
from fast.data import AbstractScale, scale_several_time_series


def get_twpf_column_names(return_ex_vars: bool = True) -> List:
    """
        Get the columns of the csv file.
    """
    names = ['Datetime', 'LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']

    if return_ex_vars:
        return names[2:]

    return names


def load_turkey_wpf_sst(data_root: str,
                        use_ex_variables: bool = True,
                        use_time_variable: bool = False,
                        time_granularity: Literal['10min', '30min', '1hour', '1day'] = 'day',
                        frequency: Literal['Y', 'ME', 'W', 'd', 'B', 'h', 'min', 's'] = 'd',
                        is_time_normalized: bool = True,
                        input_window_size: int = 96,
                        output_window_size: int = 24,
                        horizon: int = 1,
                        stride: int = 1,
                        train_ratio: float = 0.8,
                        val_ratio: float = None,
                        factor: float = 0.0001,
                        scaler: AbstractScale = None,
                        ex_scaler: AbstractScale = None):
    float_type = np.float32  # the default float type is ``float32``, you can change it to ``float64`` if needed
    device = torch.device('cpu') # the default device is ``cpu``, you can change it to ``cuda`` if needed
    column_names = get_twpf_column_names()

    csv_dir = os.path.join(data_root, r'time_series\energy_wind\kaggle_Turkey_turbine\2_mts_csv')
    csv_file = csv_dir + f'/Turkey_wind_turbine_{time_granularity}.csv'

    time_variable = column_names[0] if use_time_variable else None

    print('Loading {}'.format(csv_file))
    df = pd.read_csv(csv_file)

    sst_params: dict[str, Union[int, float, str, torch.Tensor]] = \
        {'input_window_size': input_window_size, 'output_window_size': output_window_size, 'horizon': horizon}

    target_df = df.loc[:, column_names[1]].to_frame()
    target_array = target_df.values.astype(float_type) * factor
    target_tensor = torch.tensor(target_array, device=device)
    sst_params['ts'] = target_tensor

    if scaler is not None:
        scaler = scale_several_time_series(scaler, target_tensor)

    if use_ex_variables:
        ex_df = df.loc[:, column_names[2]].to_frame()
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_params['ex_ts'] = ex_tensor

        if ex_scaler is not None:
            ex_scaler = scale_several_time_series(ex_scaler, ex_tensor)

    if time_variable is not None:
        df[time_variable] = pd.to_datetime(df[time_variable])
        time_features = TimeAsFeature(freq=frequency, is_normalized=is_time_normalized)(df[time_variable].dt)
        time_feature_tensor = torch.tensor(time_features, device=device)
        sst_params["ex_ts2"] = time_feature_tensor

    if train_ratio == 1.0:
        train_ds = SSTDataset(**sst_params)
        return (train_ds), (scaler, ex_scaler)

    if val_ratio is None:
        train_ds = SSTDataset(**sst_params, stride=stride).split(train_ratio, 'train', 'train')
        val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size']).split(train_ratio, 'val', 'val')
        return (train_ds, val_ds), (scaler, ex_scaler)

    ratio1 = train_ratio + val_ratio        # split ratio of (train + val) / test
    ratio2 = train_ratio / ratio1           # split ratio of train / (train + val)
    o_stride = sst_params['output_window_size']
    train_ds = SSTDataset(**sst_params, stride=stride).split(ratio1, 'train').split(ratio2, 'train', 'train')
    val_ds = SSTDataset(**sst_params, stride=o_stride).split(ratio1, 'train').split(ratio2, 'val', 'val')
    test_ds = SSTDataset(**sst_params, stride=o_stride).split(ratio1, 'val', 'test')

    return (train_ds, val_ds, test_ds), (scaler, ex_scaler)
