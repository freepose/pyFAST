#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading Shang_Diabetes ``STSDataset`` and ``STMDataset``.
"""
import os, sys, random
from typing import Literal, List
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch

from fast.data import Scale, StandardScale, MinMaxScale, train_test_split, time_series_scaler
from fast.data import STSDataset, STMDataset


def get_sh_diabetes(csv_files: List[str], ex_names: List[str] = None, desc: str = None) -> tuple:
    """
        Qinpei Zhao, et al.,
        Chinese diabetes datasets for data-driven machine learning, nature scientific data, 2023.
        DOI: 10.1038/s41597-023-01940-7

        columns = [Date,CGM,CGB,Blood ketone,
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
        :param ex_names: the exogenous variable names, which are input features.
        :param desc: the description of the file reading action.
        :return:
    """
    target_var_names = ['CGM']

    time_tensor_list, target_tensor_list, ex_tensor_list, ex_tensor_mask_list = [], [], [], []
    with tqdm(total=len(csv_files), leave=False, file=sys.stdout) as pbar:
        for name in csv_files:
            pbar.set_description(desc)

            df = pd.read_csv(name, index_col=None)

            df['Date'] = pd.to_datetime(df['Date'])
            base_time = df['Date'].min().normalize()
            df['month'] = df['Date'].dt.month.values
            df['day'] = df['Date'].dt.day.values
            df['timestamp'] = (df['Date'] - base_time).dt.total_seconds() / 60.0
            df['dayofweek'] = df['Date'].dt.dayofweek.values
            df['hour'] = df['Date'].dt.hour.values
            df['minute'] = df['Date'].dt.minute.values

            time_array = df[['timestamp', 'dayofweek', 'hour', 'minute']].values.astype(np.float32)
            target_array = df[target_var_names].values.reshape(-1, 1).astype(np.float32)

            time_tensor_list.append(torch.tensor(time_array))
            target_tensor_list.append(torch.tensor(target_array))
            if ex_names is not None:
                ex_array = df[ex_names].values.astype(np.float32)
                ex_array_mask = ~np.isnan(ex_array)
                ex_tensor_list.append(torch.tensor(ex_array))
                ex_tensor_mask_list.append(torch.tensor(ex_array_mask))

            pbar.update(1)

    if ex_names is not None:
        return time_tensor_list, target_tensor_list, ex_tensor_list, ex_tensor_mask_list

    return time_tensor_list, target_tensor_list


def get_csv_files(directory: str) -> List[str]:
    """
        Get the csv files in the directory.
        :param directory: the directory path.
        :return: the csv files in the ``directory``.
    """
    csv_files = [directory + '/' + name for name in os.listdir(directory) if name.endswith('.csv')]
    return csv_files


def load_sh_diabetes_stm(data_root: str,
                         disease: Literal['T1DM', 'T2DM', 'all'] = 'T1DM',
                         ds_params: dict = None,
                         split_task: Literal['intra', 'inter'] = 'inter',
                         split_ratio: float = 0.8,
                         scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load Shang_Diabetes CGM as ``STMDataset`` for mts forecasting (using exogenous data).

        The target variable is **CGM**. The time interval is 15min.
        :param data_root: the root path of the dataset.
        :param disease: the disease type, including T1DM, T2DM, and all.
        :param ds_params: the dataset parameters.
        :param split_task: the split task, including 'in' and 'inter'.
                            'intra' means training and test in different parts of a time series., i.e., traditional tsf.
                            'inter' means training set and test have non-overlapping time series, i.e., personalized.
        :param split_ratio: the split ratio.
        :param scaler: the scaler for the dataset.
        :return: the training and validation dataset, and the scaler.
    """
    columns = [
        'Date', 'CGM', 'CGB', 'Blood ketone', 'Dietary intake', 'Bolus insulin', 'Basal insulin',
        'Insulin dose s.c. id:0 medicine', 'Insulin dose s.c. id:0 dosage',
        'Insulin dose s.c. id:1 medicine', 'Insulin dose s.c. id:1 dosage',
        'Non-insulin id:0 medicine', 'Non-insulin id:0 dosage',
        'Non-insulin id:1 medicine', 'Non-insulin id:1 dosage',
        'Non-insulin id:2 medicine', 'Non-insulin id:2 dosage',
        'Insulin dose i.v. id:0 medicine', 'Insulin dose i.v. id:0 dosage',
        'Insulin dose i.v. id:1 medicine', 'Insulin dose i.v. id:1 dosage',
        'Insulin dose i.v. id:2 medicine', 'Insulin dose i.v. id:2 dosage']
    ex_columns = ['Dietary intake', 'Bolus insulin', 'Basal insulin'] + columns[8::2]

    sh_diabetes_dir = data_root + '/time_series/disease/Shanghai_T1DM_T2DM/4_multiple_uts'
    t1_dir = sh_diabetes_dir + '/Shanghai_T1DM'
    t2_dir = sh_diabetes_dir + '/Shanghai_T2DM'

    if disease == 'T1DM':
        csv_files = get_csv_files(t1_dir)
    elif disease == 'T2DM':
        csv_files = get_csv_files(t2_dir)
    else:  # 'all' case
        csv_files = get_csv_files(t1_dir) + get_csv_files(t2_dir)

    random.shuffle(csv_files)

    time_tensor_list, target_tensor_list, ex_tensor_list, ex_tensor_mask_list = \
        get_sh_diabetes(csv_files, ex_columns, 'Loading {}'.format(disease))

    if split_task == 'intra':
        ds_params.update({'split_ratio': split_ratio})
        train_ds = STMDataset(target_tensor_list, None, ex_tensor_list, ex_tensor_mask_list, **ds_params, split='train')
        ds_params.update({'stride': ds_params['output_window_size']})
        val_ds = STMDataset(target_tensor_list, None, ex_tensor_list, ex_tensor_mask_list, **ds_params, split='val')
        scaler = time_series_scaler(target_tensor_list, scaler)
    else:
        split_pos = int(len(csv_files) * split_ratio)
        data = list(zip(time_tensor_list, target_tensor_list, ex_tensor_list, ex_tensor_mask_list))
        train_data, val_data = data[:split_pos], data[split_pos:]
        train_time, train_target, train_ex, train_ex_mask = zip(*train_data)
        val_time, val_target, val_ex, val_ex_mask = zip(*val_data)

        ds_params.update({'split_ratio': 1.0, 'split': 'train'})
        train_ds = STMDataset(train_target, None, train_ex, train_ex_mask, stride=1, **ds_params)
        val_ds = STMDataset(val_target, None, val_ex, val_ex_mask, stride=ds_params['output_window_size'], **ds_params)

        scaler = time_series_scaler(train_target, scaler)

    return (train_ds, val_ds), (scaler, None)
