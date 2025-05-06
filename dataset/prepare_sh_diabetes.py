#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading Shanghai Diabetes as ``SMTDataset``.
"""

import os, sys, random
from typing import Literal, List
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from fast.data import AbstractScale, StandardScale, MinMaxScale, scale_several_time_series
from fast.data import SMTDataset
from dataset.time_feature import TimeAsFeature


def get_column_names(return_ex_vars: bool = True) -> List:
    """
        Get the columns of the csv file.
    """
    names = [
        'Date', 'CGM', 'CGB', 'Blood ketone', 'Dietary intake', 'Bolus insulin', 'Basal insulin',
        'Insulin dose s.c. id:0 medicine', 'Insulin dose s.c. id:0 dosage',
        'Insulin dose s.c. id:1 medicine', 'Insulin dose s.c. id:1 dosage',
        'Non-insulin id:0 medicine', 'Non-insulin id:0 dosage',
        'Non-insulin id:1 medicine', 'Non-insulin id:1 dosage',
        'Non-insulin id:2 medicine', 'Non-insulin id:2 dosage',
        'Insulin dose i.v. id:0 medicine', 'Insulin dose i.v. id:0 dosage',
        'Insulin dose i.v. id:1 medicine', 'Insulin dose i.v. id:1 dosage',
        'Insulin dose i.v. id:2 medicine', 'Insulin dose i.v. id:2 dosage']

    if return_ex_vars:
        return ['Dietary intake', 'Bolus insulin', 'Basal insulin'] + names[8::2]

    return names


def get_sh_diabetes(csv_files: List[str],
                    factor: float = 1.,
                    ex_vars: List[str] = None,
                    use_time_features: bool = False,
                    desc: str = None) -> List:
    """
        Qinpei Zhao, et al.,
        Chinese diabetes datasets for data-driven machine learning, nature scientific data, 2023.
        DOI: 10.1038/s41597-023-01940-7

        columns = [Date, CGM, CGB, Blood ketone,
            Dietary intake,
            Bolus insulin,
            Basal insulin,
            Insulin dose s.c. id:0 medicine,
            Insulin dose s.c. id:0 dosage,
            Insulin dose s.c. id:1 medicine,
            Insulin dose s.c. id:1 dosage,
            Non-insulin id:0 medicine,
            Non-insulin id:0 dosage,
            Non-insulin id:1 medicine,
            Non-insulin id:1 dosage,
            Non-insulin id:2 medicine,
            Non-insulin id:2 dosage,
            Insulin dose i.v. id:0 medicine,
            Insulin dose i.v. id:0 dosage,
            Insulin dose i.v. id:1 medicine,
            Insulin dose i.v. id:1 dosage,
            Insulin dose i.v. id:2 medicine,
            Insulin dose i.v. id:2 dosage]

        :param csv_files: the csv file list of absolute paths.
        :param factor: the factor to scale the wind power values.
        :param ex_vars: the exogenous variable names, which are input features.
        :param use_time_features: whether to use time features.
        :param desc: the description of the file reading action.
        :return:
    """
    ret_list = []
    with tqdm(total=len(csv_files), leave=False, file=sys.stdout) as pbar:
        for name in csv_files:
            pbar.set_description(desc)

            df = pd.read_csv(name, index_col=None)
            tensor_list = []

            target_array = df['CGM'].values.reshape(-1, 1).astype(np.float32) * factor
            target_tensor = torch.tensor(target_array)
            tensor_list.append(target_tensor)

            if ex_vars is not None:
                ex_array = df[ex_vars].values.astype(np.float32)
                ex_array_mask = ~np.isnan(ex_array)
                ex_tensor = torch.tensor(ex_array)
                ex_tensor_mask = torch.tensor(ex_array_mask)
                tensor_list.append(ex_tensor)
                tensor_list.append(ex_tensor_mask)

            if use_time_features:
                df['Date'] = pd.to_datetime(df['Date'])
                time_dt = df['Date'].dt
                # Need a mapping from ``freq`` of csv file to ``freq`` of TimeAsFeature.
                time_feature_array = TimeAsFeature(freq='d', is_normalized=True)(time_dt)
                time_feature_array = time_feature_array.astype(np.float32)
                time_feature_tensor = torch.tensor(time_feature_array)
                tensor_list.append(time_feature_tensor)

            ret_list.append(tensor_list)
            pbar.update(1)

    ret_list2 = list(zip(*ret_list))

    return ret_list2


def load_sh_diabetes_smt(data_root: str,
                         disease: Literal['T1DM', 'T2DM', 'all'] = 'T1DM',
                         ex_vars: List[str] = None,
                         use_time_features: bool = False,
                         split_ratio: float = 0.8,
                         input_window_size: int = 10,
                         output_window_size: int = 1,
                         horizon: int = 1,
                         stride: int = 1,
                         factor: float = 1.,
                         scaler: AbstractScale = None,
                         ex_scaler: AbstractScale = None) -> tuple[tuple, tuple]:
    """
        Load Shang Diabetes CGM as ``SMTDataset`` for time series forecasting (using exogenous data).

        Use inter split task.

        The target variable is **CGM**. The time interval is 15min.
        :param data_root: the root path of the dataset.
        :param disease: the disease type, including T1DM, T2DM, and all.
        :param ex_vars: whether to use exogenous time series data.
                        The values are one, several or all in [Date, CGM, CGB, Blood ketone,
                        Dietary intake,
                        Bolus insulin,
                        Basal insulin,
                        Insulin dose s.c. id:0 medicine,
                        Insulin dose s.c. id:0 dosage,
                        Insulin dose s.c. id:1 medicine,
                        Insulin dose s.c. id:1 dosage,
                        Non-insulin id:0 medicine,
                        Non-insulin id:0 dosage,
                        Non-insulin id:1 medicine,
                        Non-insulin id:1 dosage,
                        Non-insulin id:2 medicine,
                        Non-insulin id:2 dosage,
                        Insulin dose i.v. id:0 medicine,
                        Insulin dose i.v. id:0 dosage,
                        Insulin dose i.v. id:1 medicine,
                        Insulin dose i.v. id:1 dosage,
                        Insulin dose i.v. id:2 medicine,
                        Insulin dose i.v. id:2 dosage]
        :param use_time_features: whether to use time features, default is False.
        :param split_ratio: the split ratio.
        :param input_window_size: the input window size, default is 10.
        :param output_window_size: the output window size, default is 1.
        :param horizon: the distance between two input window and output window, default is 1.
        :param stride: the distance between two consecutive (input / output) windows, default is 1.
        :param factor: the factor on the target variable (i.e., active power), default is 1.
        :param scaler: the global scaler class for the target datasets, default is Scale().
        :param ex_scaler: the global scaler class for the exogenous datasets, default is Scale().
        :return: tuple of (train_ds, val_ds), (scaler, None).
    """
    assert disease in ['T1DM', 'T2DM', 'all'], f'Invalid disease type: {disease}'
    assert 0 < split_ratio <= 1.0, f'Invalid split ratio: {split_ratio}'

    csv_dir = os.path.join(data_root, 'time_series/disease/Shanghai_T1DM_T2DM/4_multiple_uts')
    t1_csv_dir = os.path.join(csv_dir, 'Shanghai_T1DM')
    t2_csv_dir = os.path.join(csv_dir, 'Shanghai_T2DM')

    # Get CSV files based on disease type
    if disease == 'T1DM':
        csv_files = [t1_csv_dir + '/' + name for name in os.listdir(t1_csv_dir) if name.endswith('.csv')]
    elif disease == 'T2DM':
        csv_files = [t2_csv_dir + '/' + name for name in os.listdir(t2_csv_dir) if name.endswith('.csv')]
    else:  # 'all' case
        csv_files = [t1_csv_dir + '/' + name for name in os.listdir(t1_csv_dir) if name.endswith('.csv')]
        csv_files += [t2_csv_dir + '/' + name for name in os.listdir(t2_csv_dir) if name.endswith('.csv')]

    if split_ratio == 1.0:
        data = get_sh_diabetes(csv_files, factor, ex_vars, use_time_features, desc='Loading training files...')

        smt_params = {
            'ts': data[0],
            'ex_ts': data[1] if ex_vars is not None else None,
            'ex_ts_mask': data[2] if ex_vars is not None else None,
            'ex_ts2': data[-1] if use_time_features else None,
            'input_window_size': input_window_size,
            'output_window_size': output_window_size,
            'horizon': horizon,
        }

        if scaler is not None and type(scaler) != type(AbstractScale()):
            scaler = scale_several_time_series(data[0], scaler)

        if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(AbstractScale()):
            ex_scaler = scale_several_time_series(data[1], ex_scaler)

        train_ds = SMTDataset(**smt_params, stride=stride)
        return (train_ds, None), (scaler, ex_scaler)

    random.shuffle(csv_files)
    split_position = int(len(csv_files) * split_ratio)
    train_files, val_files = csv_files[:split_position], csv_files[split_position:]

    train_data = get_sh_diabetes(train_files, factor, ex_vars, use_time_features, desc='Loading training files...')
    val_data = get_sh_diabetes(val_files, factor, ex_vars, use_time_features, desc='Loading validation files...')

    train_smt_params = {
        'ts': train_data[0],
        'ex_ts': train_data[1] if ex_vars is not None else None,
        'ex_ts_mask': train_data[2] if ex_vars is not None else None,
        'ex_ts2': train_data[-1] if use_time_features else None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    val_smt_params = {
        'ts': val_data[0],
        'ex_ts': val_data[1] if ex_vars is not None else None,
        'ex_ts_mask': val_data[2] if ex_vars is not None else None,
        'ex_ts2': val_data[-1] if use_time_features else None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    if scaler is not None and type(scaler) != type(AbstractScale()):
        scaler = scale_several_time_series(train_data[0], scaler)

    if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(AbstractScale()):
        ex_scaler = scale_several_time_series(train_data[1], ex_scaler)

    train_ds = SMTDataset(**train_smt_params, stride=stride)
    val_ds = SMTDataset(**val_smt_params, stride=output_window_size)

    return (train_ds, val_ds), (scaler, ex_scaler)
