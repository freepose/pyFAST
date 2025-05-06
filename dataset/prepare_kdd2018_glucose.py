#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare KDD2018 Glucose dataset for training or evaluation.
"""

import os, sys, json
import random
from typing import Literal

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

from fast.data import AbstractScale, StandardScale, MinMaxScale, scale_several_time_series
from fast.data import SMTDataset


def load_kdd2018_glucose_smt(data_root: str,
                             split_ratio: float = 0.8,
                             input_window_size: int = 5 * 12,
                             output_window_size: int = 6,
                             horizon: int = 1,
                             stride: int = 1,
                             factor: float = 1.,
                             scaler: AbstractScale = None) -> tuple[tuple, tuple]:
    """
        Load KDD 2018 glucose datasets.

        The data file is in json format.
        The time interval is **5 minutes**.
        Use inter split task.

        :param data_root: the root directory of the whole datasets.
        :param split_ratio: the ratio of training set. Split according to csv files (inter-patient).
        :param input_window_size: the input window size, default is 5 * 12.
        :param output_window_size: the output window size, default is 6.
        :param horizon: the distance between two input window and output window, default is 1.
        :param stride: the distance between two consecutive (input / output) windows, default is 1.
        :param factor: the factor on the target variable (i.e., active power), default is 1.
        :param scaler: the global scaler for the datasets, default is Scale().
        :return: train and validation datasets, and global scalers for target and exogenous time series.
    """
    json_file = data_root + 'time_series/disease/kdd2018_glucose_forecasting/kdd2018_cgm_data.json'

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    with tqdm(total=len(json_data), leave=False, file=sys.stdout) as pbar:
        cgm_uts_list = []
        for pid, cgm_list in json_data.items():
            pbar.set_description('Loading {}'.format(pid))

            cgm_df = pd.DataFrame(data=cgm_list)
            cgm_df_smooth = cgm_df.rolling(window=5, min_periods=1).mean()
            cgm_df_smooth = cgm_df_smooth.clip(40, 400)

            if cgm_df_smooth.shape[0] < 100:
                continue

            if cgm_df_smooth.nunique().values[0] == 1:
                continue

            # dff_abs = cgm_df_smooth.diff().abs().fillna(0)

            cgm_uts_list.append(torch.tensor(cgm_df_smooth.values.reshape(-1, 1).astype(np.float32)) * factor)
            pbar.update(1)

    if split_ratio == 1.0:
        smt_params = {
            'ts': cgm_uts_list,
            'ex_ts': None,
            'ex_ts_mask': None,
            'ex_ts2': None,
            'input_window_size': input_window_size,
            'output_window_size': output_window_size,
            'horizon': horizon,
        }

        if scaler is not None and type(scaler) != type(AbstractScale()):
            scaler = scale_several_time_series(cgm_uts_list, scaler)

        train_ds = SMTDataset(**smt_params, stride=stride)
        return (train_ds, None), (scaler, None)

    random.shuffle(cgm_uts_list)
    split_position = int(len(cgm_uts_list) * split_ratio)
    train_data, val_data = cgm_uts_list[:split_position], cgm_uts_list[split_position:]

    train_smt_params = {
        'ts': train_data,
        'ex_ts': None,
        'ex_ts_mask': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    val_smt_params = {
        'ts': val_data,
        'ex_ts': None,
        'ex_ts_mask': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    if scaler is not None and type(scaler) != type(AbstractScale()):
        scaler = scale_several_time_series(train_data, scaler)

    train_ds = SMTDataset(**train_smt_params, stride=stride)
    val_ds = SMTDataset(**val_smt_params, stride=output_window_size)

    return (train_ds, val_ds), (scaler, None)
