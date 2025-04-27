#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading greek-wind-energy-forecasting as ``STMDataset``.
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


def get_gwpf_column_names(return_ex_vars: bool = True) -> List:
    """
        Get the columns of the csv file.
    """
    names = ['time', 'power(MW)', 'airTemperature', 'cloudCover', 'gust', 'humidity',
             'precipitation', 'pressure', 'visibility', 'windDirection', 'windSpeed']

    if return_ex_vars:
        return names[2:]

    return names


def get_greek_wind(csv_files: List[str],
                   factor: float = 1.,
                   ex_vars: List[str] = None,
                   use_time_features: bool = False,
                   desc: str = None) -> List:
    """
        Read the Greek wind power dataset from csv files.

        https://github.com/intelligence-csd-auth-gr/greek-solar-wind-energy-forecasting/tree/main/resources/raw_data

        The locations of the wind turbines are:
        id1	representative_lat	representative_lon
        32947	38.774952	20.993431
        33332	37.485296	23.155247
        33466	38.478476	23.317671
        33629	37.534464	22.597457
        33647	41.056586	26.013658
        33651	39.774893	20.548365
        33652	39.836571	20.519751
        33714	38.317112	22.931187
        33804	38.313572	22.581566
        33805	37.691283	22.459475
        33815	37.396455	22.362301
        33827	37.475152	23.926183
        33837	38.233498	23.496811
        34105	38.29736	20.512369
        34391	38.244536	23.11726
        34398	41.153016	25.835738
        34443	41.286367	25.904589
        36876	38.769349	21.232926

        columns = [time, power(MW), airTemperature, cloudCover, gust, humidity,
        precipitation, pressure, visibility, windDirection, windSpeed]

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

            target_array = df['power(MW)'].values.reshape(-1, 1).astype(np.float32) * factor
            target_tensor = torch.tensor(target_array)
            tensor_list.append(target_tensor)

            if ex_vars is not None:
                ex_array = df[ex_vars].values.astype(np.float32)
                ex_tensor = torch.tensor(ex_array)
                tensor_list.append(ex_tensor)

            if use_time_features:
                df['time'] = pd.to_datetime(df['time'])
                time_dt = df['time'].dt
                # Need a mapping from ``freq`` of csv file to ``freq`` of TimeAsFeature.
                time_feature_array = TimeAsFeature(freq='d', is_normalized=True)(time_dt)
                time_feature_array = time_feature_array.astype(np.float32)
                time_feature_tensor = torch.tensor(time_feature_array)
                tensor_list.append(time_feature_tensor)

            ret_list.append(tensor_list)
            pbar.update(1)

    ret_list2 = list(zip(*ret_list))

    return ret_list2


def load_greece_wpf_smt(data_root: str,
                        freq: Literal['1hour', '6hour', '12hour', '1day'] = '1hour',
                        ex_vars: List[str] = None,
                        use_time_features: bool = False,
                        split_task: Literal['intra', 'inter'] = 'inter',
                        split_ratio: float = 0.8,
                        input_window_size: int = 10,
                        output_window_size: int = 1,
                        horizon: int = 1,
                        stride: int = 1,
                        factor: float = 1.,
                        scaler: Scale = None,
                        ex_scaler: Scale = None) -> Tuple:
    """
        Load Greek wind power dataset as ``SMTDataset`` for time series forecasting (using exogenous data).

        The Unit of wind power is **MW**.

        :param data_root: the root directory of the whole datasets.
        :param freq: the frequency of the time series, ['1hour', '6hour', '12hour', '1day'].
        :param ex_vars: whether to use exogenous time series data.
                        The values are one, several or all in ['airTemperature', 'cloudCover', 'gust', 'humidity',
                            'precipitation', 'pressure', 'visibility', 'windDirection', 'windSpeed']
        :param use_time_features: whether to use time features, default is False.
        :param split_task: the split task, including 'intra' and 'inter'.
                    'intra' means training and test in different parts of the same time series, i.e., traditional tsf.
                    'inter' means training and test set are from different time series, i.e., personalized tsf.
        :param split_ratio: the ratio of training set. Split according to csv files (or turbines).
        :param input_window_size: the input window size, default is 10.
        :param output_window_size: the output window size, default is 1.
        :param horizon: the distance between two input window and output window, default is 1.
        :param stride: the distance between two consecutive (input / output) windows, default is 1.
        :param factor: the factor on the target variable (i.e., active power), default is 1.
        :param scaler: the global scaler class for the target datasets, default is Scale().
        :param ex_scaler: the global scaler class for the exogenous datasets, default is Scale().
        :return: tuple of (train_ds, val_ds), (scaler, None).
    """
    assert freq in ['1hour', '6hour', '12hour', '1day'], f'Invalid frequency: {freq}'
    assert split_task in ['intra', 'inter'], f'Invalid split task: {split_task}'
    assert 0 < split_ratio <= 1.0, f'Invalid split ratio: {split_ratio}'

    csv_dir = data_root + r'time_series/energy_wind/greek-wind-energy-forecasting/4_multiple_uts/' + freq
    csv_files = [csv_dir + '/' + name for name in os.listdir(csv_dir) if name.endswith('.csv')]

    if split_task == 'intra':
        data = get_greek_wind(csv_files, factor, ex_vars, use_time_features, desc='Loading files...')

        smt_params = {
            'ts': data[0],
            'ex_ts': data[1] if ex_vars is not None else None,
            'ex_ts_mask': None,
            'ex_ts2': data[-1] if use_time_features else None,
            'input_window_size': input_window_size,
            'output_window_size': output_window_size,
            'horizon': horizon,
        }

        if scaler is not None and type(scaler) != type(Scale()):
            scaler = time_series_scaler(data[0], scaler)

        if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(Scale()):
            ex_scaler = time_series_scaler(data[1], ex_scaler)

        if split_ratio == 1.0:
            train_ds = SMTDataset(**smt_params, stride=stride)
            return (train_ds, None), (scaler, ex_scaler)

        train_ds = SMTDataset(**smt_params, stride=stride).split(split_ratio, 'train', 'train')
        val_ds = SMTDataset(**smt_params, stride=output_window_size).split(split_ratio, 'val', 'val')

        return (train_ds, val_ds), (scaler, ex_scaler)

    else: # elif split_task == 'inter':

        if split_ratio == 1.0:
            data = get_greek_wind(csv_files, factor, ex_vars, use_time_features, desc='Loading training files...')

            smt_params = {
                'ts': data[0],
                'ex_ts': data[1] if ex_vars is not None else None,
                'ex_ts_mask': None,
                'ex_ts2': data[-1] if use_time_features else None,
                'input_window_size': input_window_size,
                'output_window_size': output_window_size,
                'horizon': horizon,
            }

            if scaler is not None and type(scaler) != type(Scale()):
                scaler = time_series_scaler(data[0], scaler)

            if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(Scale()):
                ex_scaler = time_series_scaler(data[1], ex_scaler)

            train_ds = SMTDataset(**smt_params, stride=stride)
            return (train_ds, None), (scaler, ex_scaler)

        random.shuffle(csv_files)
        split_position = int(len(csv_files) * split_ratio)
        train_files, val_files = csv_files[:split_position], csv_files[split_position:]

        train_data = get_greek_wind(train_files, factor, ex_vars, use_time_features, desc='Loading training files...')
        val_data = get_greek_wind(val_files, factor, ex_vars, use_time_features, desc='Loading validation files...')

        train_smt_params = {
            'ts': train_data[0],
            'ex_ts': train_data[1] if ex_vars is not None else None,
            'ex_ts_mask': None,
            'ex_ts2': train_data[-1] if use_time_features else None,
            'input_window_size': input_window_size,
            'output_window_size': output_window_size,
            'horizon': horizon,
        }

        val_smt_params = {
            'ts': val_data[0],
            'ex_ts': val_data[1] if ex_vars is not None else None,
            'ex_ts_mask': None,
            'ex_ts2': val_data[-1] if use_time_features else None,
            'input_window_size': input_window_size,
            'output_window_size': output_window_size,
            'horizon': horizon,
        }

        if scaler is not None and type(scaler) != type(Scale()):
            scaler = time_series_scaler(train_data[0], scaler)

        if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(Scale()):
            ex_scaler = time_series_scaler(train_data[1], ex_scaler)

        train_ds = SMTDataset(**train_smt_params, stride=stride, mark='train')
        val_ds = SMTDataset(**val_smt_params, stride=output_window_size, mark='val')

        return (train_ds, val_ds), (scaler, ex_scaler)
