#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare dataset for training or evaluation.
"""
import os, copy, sys, json
import random
from typing import Type, Literal

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch

from fast.data import UTSDataset, Scale, StandardScale, MinMaxScale, train_test_split, STSDataset
from dataset.read_data import get_xmcdc, get_sh_diabetes, get_greek_wind, get_kddcup2022_sdwpf_list
from dataset.read_data import get_nacom_2024_battery, get_nenergy_2019_battery


def time_series_scaler(ts: torch.Tensor or list[torch.Tensor], scaler: Scale()) -> Scale:
    """
        Scale the datasets.
        :param ts: the list of time series.
        :param scaler: the scaler.
        :return: the scaled time series.
    """
    if type(scaler) == type(Scale()):
        return Scale()

    scale_ts = ts
    if isinstance(ts, list):
        scale_ts = torch.cat(ts, dim=0)
    scaler = copy.deepcopy(scaler).fit(scale_ts)
    del scale_ts

    return scaler


"""
    Disease datasets.
"""


def load_xmcdc_cases(data_root: str = '../dataset/xmcdc/',
                     frequency: Literal['daily', 'weekly'] = 'daily',
                     ds_params: dict = None,
                     split_ratio: float = 0.8,
                     use_ex_ts: bool = False,
                     scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load XMCDC disease outpatient case count for multiple uts forecasting.
        :param data_root: the root directory of the whole datasets.
        :param frequency: the frequency of the time series, either 'daily' or 'weekly'.
        :param ds_params: the common dataset parameters for train and test ``UTSDataset``.
        :param split_ratio: the ratio of training set. A time series is split into train and test.
        :param use_ex_ts: whether to use time as exogenous factor, default is False.
        :param scaler: the global scaler target time series and exogenous time series, default is Scale().
        :return: train and validation datasets, and global scalers for target and exogenous time series.
    """
    csv_file = data_root + r'{}_outpatients_2011_2020.csv'.format(frequency)
    disease_key_dict = {'手足口病': 'BSI_厦门手足口病', '肝炎': 'BSI_厦门肝炎', '其他感染性腹泻': 'BSI_厦门腹泻'}

    df = pd.read_csv(csv_file)
    weather_df = df.loc[:, '平均温度':'最低风速']
    weather_array = weather_df.values.astype(np.float32)

    uts_array_list, ex_ts_list = [], []
    for disease, key in disease_key_dict.items():
        disease_df = df[disease]
        disease_array = disease_df.values.reshape(-1, 1).astype(np.float32)

        bsi_df = df[[col for col in df.columns if key in col]]
        bsi_array = bsi_df.values.astype(np.float32)

        uts_array_list.append(disease_array)
        ex_ts_list.append(np.concatenate([weather_array, bsi_array], axis=1))

    uts_tensor_list = [torch.tensor(arr) for arr in uts_array_list]

    global_scaler = time_series_scaler(uts_tensor_list, scaler)
    global_ex_scaler, ex_ts_tensor_list = None, None
    if use_ex_ts:
        ex_ts_tensor_list = [torch.tensor(arr) for arr in ex_ts_list]
        global_ex_scaler = time_series_scaler(ex_ts_tensor_list, scaler)

    ds_params.update({'split_ratio': split_ratio, 'split': 'train'})
    train_ds = UTSDataset(uts_tensor_list, ex_ts=ex_ts_tensor_list, **ds_params)
    ds_params.update({'split_ratio': split_ratio, 'split': 'val'})
    val_ds = UTSDataset(uts_tensor_list, ex_ts=ex_ts_tensor_list, **ds_params)

    return (train_ds, val_ds), (global_scaler, global_ex_scaler)


def load_sh_diabetes(data_root: str,
                     ds_params: dict,
                     split_ratio: float = 0.8,
                     use_ex_ts: bool = False,
                     scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load Shanghai diabetes datasets for model training and evaluation.

        The data fields are: ['Date','CGM (mg/ dl)']. The time interval is **15 minutes**.

        :param data_root: the root directory of the whole datasets.
        :param ds_params: []
        :param split_ratio: the ratio of training set. Split according to csv files (inter-patient).
        :param use_ex_ts: whether to use time as exogenous factor, default is False.
        :param scaler: the global scaler class for the datasets, default is Scale().
        :return:
    """
    sh_diabetes_data_root = data_root + 'time_series/disease/Shanghai_T1DM_T2DM/4_multiple_uts'

    t_li1, cgm_li1 = get_sh_diabetes(sh_diabetes_data_root, 'T1')
    t_li2, cgm_li2 = get_sh_diabetes(sh_diabetes_data_root, 'T2')
    time_array_list, cgm_array_list = t_li1 + t_li2, cgm_li1 + cgm_li2

    paired_list = list(zip(time_array_list, cgm_array_list))
    random.shuffle(paired_list)
    time_array_list, cgm_array_list = zip(*paired_list)
    time_array_list, cgm_array_list = list(time_array_list), list(cgm_array_list)

    split_position = int(len(time_array_list) * split_ratio)

    train_time_list, val_time_list = time_array_list[:split_position], time_array_list[split_position:]
    train_cgm_list, val_cgm_list = cgm_array_list[:split_position], cgm_array_list[split_position:]

    train_time_list = [torch.tensor(arr) for arr in train_time_list]
    train_cgm_list = [torch.tensor(arr) for arr in train_cgm_list]
    val_time_list = [torch.tensor(arr) for arr in val_time_list]
    val_cgm_list = [torch.tensor(arr) for arr in val_cgm_list]

    train_ex_ts, val_ex_ts = None, None
    global_scaler, global_ex_scaler = time_series_scaler(train_cgm_list, scaler), None
    if use_ex_ts:
        train_ex_ts, val_ex_ts = train_time_list, val_time_list
        global_ex_scaler = time_series_scaler(train_ex_ts, scaler)

    ds_params.update({'split_ratio': 1., 'split': 'train'})
    train_ds = UTSDataset(train_cgm_list, ex_ts=train_ex_ts, **ds_params, stride=1)
    val_ds = UTSDataset(val_cgm_list, ex_ts=val_ex_ts, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_scaler, global_ex_scaler)


def load_kdd2018_glucose(data_root: str,
                         ds_params: dict,
                         split_ratio: float = 0.8,
                         scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load KDD 2018 glucose datasets for model training and evaluation.

        The data file is in json format.
        The time interval is **5 minutes**.

        :param data_root: the root directory of the whole datasets.
        :param ds_params: the dataset parameters.
        :param split_ratio: the ratio of training set. Split according to csv files (inter-patient).
        :param scaler: the global scaler for the datasets, default is Scale().
        :return: train and validation datasets, and global scalers for target and exogenous time series.
    """
    json_file = data_root + 'time_series/disease/kdd2018_glucose_forecasting/kdd2018_unprocessed_cgm_data.json'

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

            cgm_uts_list.append(cgm_df_smooth.values.reshape(-1, 1).astype(np.float32))
            pbar.update(1)

    random.shuffle(cgm_uts_list)
    split_position = int(len(cgm_uts_list) * split_ratio)
    train_cgm_list, val_cgm_list = cgm_uts_list[:split_position], cgm_uts_list[split_position:]

    train_cgm_list = [torch.tensor(arr) for arr in train_cgm_list]
    val_cgm_list = [torch.tensor(arr) for arr in val_cgm_list]

    global_scaler = time_series_scaler(train_cgm_list, scaler)

    ds_params.update({'split_ratio': 1., 'split': 'train'})
    train_ds = UTSDataset(train_cgm_list, **ds_params, stride=1)
    val_ds = UTSDataset(val_cgm_list, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_scaler, None)


"""
    Energy datasets (battery).
"""


def load_nenergy_2019_battery(data_root: str,
                              ds_params: dict,
                              target: Literal['RUL(cycle)', 'SoH', 'PCL'] = 'SoH',
                              split_ratio: float = 0.8,
                              use_features: bool = False,
                              scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load Nature Energy 2019 battery cycle life datasets for model training and evaluation.

        :param data_root: the root directory of the whole datasets.
        :param ds_params: the dataset parameters.
        :param target: the target column name, is in ['RUL(cycle)', 'SoH', 'PCL'].
        :param split_ratio: the ratio of training set. Split according to csv files (inter-battery).
        :param use_features: whether to use features in the datasets.
        :param scaler: the global scaler for the datasets, default is Scale().
        :return: train and validation datasets, and global scalers for SoH and features.
    """
    nenergy_data_root = data_root + 'time_series/energy_battery/nenergy_2019_battery_cycle_life/02_ts_cycle/tiv2023_deephpm/'
    csv_file_list = [nenergy_data_root + n for n in sorted(os.listdir(nenergy_data_root)) if n.endswith('.csv')]

    random.shuffle(csv_file_list)
    split_position = int(len(csv_file_list) * split_ratio)
    train_csv_names, val_csv_names = csv_file_list[:split_position], csv_file_list[split_position:]

    train_ts_list, train_feature_list = get_nenergy_2019_battery(train_csv_names, target, use_smooth=False)
    val_ts_list, val_feature_list = get_nenergy_2019_battery(val_csv_names, target, use_smooth=False)

    train_ts_list = [torch.tensor(arr) for arr in train_ts_list]
    train_feature_list = [torch.tensor(arr) for arr in train_feature_list]
    val_ts_list = [torch.tensor(arr) for arr in val_ts_list]
    val_feature_list = [torch.tensor(arr) for arr in val_feature_list]

    train_ex_ts, val_ex_ts = None, None
    global_soh_scaler, global_feature_scaler = time_series_scaler(train_ts_list, scaler), None
    if use_features:
        train_ex_ts, val_ex_ts = train_feature_list, val_feature_list
        global_feature_scaler = time_series_scaler(train_ex_ts, scaler)

    ds_params.update({'split_ratio': 1., 'split': 'train'})
    train_ds = UTSDataset(train_ts_list, ex_ts=train_ex_ts, **ds_params, stride=1)
    val_ds = UTSDataset(val_ts_list, ex_ts=val_ex_ts, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_soh_scaler, global_feature_scaler)


def load_nacom_2024_battery(data_root: str,
                            scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load Nature community 2024 battery cycle life datasets for pytorch model training and evaluation.
        :param data_root: the root directory of the whole datasets.
        :param torch_float_type: the torch float type of tensors.
        :param scaler: the global scaler for the datasets, default is Scale().
        :return:
    """
    nacom2024_data_root = data_root + 'time_series/energy_battery/nacom_2024_battery_recycle/2_preprocess/'
    arrays = get_nacom_2024_battery(nacom2024_data_root, 'NMC2.1', 'all')
    feature_tensor, condition_tensor = [torch.tensor(arr) for arr in arrays]

    train_feats, train_cond, val_feats, val_cond = train_test_split((feature_tensor, condition_tensor), 0.8, True)

    global_feature_scaler = copy.deepcopy(scaler).fit(train_feats)  # U1 - U21
    global_condition_scaler = copy.deepcopy(scaler).fit(train_cond)  # ['SoC', 'SoH']

    window_size = 1
    sts_params = {
        'input_window_size': window_size,
        'output_window_size': window_size,
        'horizon': 1 - window_size,
        'split': 'train',
        'split_ratio': 1.,
    }

    train_ds = STSDataset(train_feats, None, train_cond, **sts_params, stride=1)
    val_ds = STSDataset(val_feats, None, val_cond, **sts_params, stride=sts_params['output_window_size'])

    return (train_ds, val_ds), (global_feature_scaler, global_condition_scaler)


"""
    Energy datasets (wind).
"""


def load_greek_wind(data_root: str,
                    frequency: Literal['1hour', '6hour', '12hour', '1day'],
                    ds_params: dict,
                    power_factor: float = 1.,
                    split_ratio: float = 0.8,
                    use_ex_ts: bool = False,
                    scaler: Scale = Scale(),
                    ex_scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load Greek wind power datasets for model training and evaluation.

        The Unit of wind power is **MW**.

        The data fields are: ['time', 'power(MW)'', 'airTemperature', 'cloudCover', 'gust', 'humidity', 'precipitation',
        'pressure', 'visibility', 'windDirection', 'windSpeed']. The time interval is **1 hour**.

        :param data_root: the root directory of the whole datasets.
        :param frequency: the frequency of the time series, ['1hour', '6hour', '12hour', '1day'].
        :param ds_params: the dataset parameters.
        :param power_factor: the power factor, default is 1.
        :param split_ratio: the ratio of training set. Split according to csv files (or turbines).
        :param use_ex_ts: whether to use time as exogenous factor, default is False.
        :param scaler: the global scaler class for the target datasets, default is Scale().
        :param ex_scaler: the global scaler class for the exogenous datasets, default is Scale().
        :return:
    """
    data_root = data_root + r'time_series/energy_wind/greek-wind-energy-forecasting/4_multiple_uts'

    data = get_greek_wind(data_root, frequency, power_factor)
    data = list(zip(*data))
    random.shuffle(data)
    split_pos = int(len(data) * split_ratio)  # file numbers

    train_data, val_data = data[:split_pos], data[split_pos:]
    train_time_list, train_ts_list, train_ex_list = list(zip(*train_data))
    val_time_list, val_ts_list, val_ex_list = list(zip(*val_data))

    train_ts_list = [torch.tensor(arr) for arr in train_ts_list]
    val_ts_list = [torch.tensor(arr) for arr in val_ts_list]
    train_ex_list = [torch.tensor(arr) for arr in train_ex_list]
    val_ex_list = [torch.tensor(arr) for arr in val_ex_list]

    train_ex_ts, val_ex_ts = None, None
    global_scaler, global_ex_scaler = time_series_scaler(train_ts_list, scaler), None
    if use_ex_ts:
        train_ex_ts, val_ex_ts = train_ex_list, val_ex_list
        global_ex_scaler = time_series_scaler(train_ex_ts, ex_scaler)

    ds_params.update({'split_ratio': 1., 'split': 'train'})
    train_ds = UTSDataset(train_ts_list, ex_ts=train_ex_ts, **ds_params, stride=1)
    val_ds = UTSDataset(val_ts_list, ex_ts=val_ex_ts, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_scaler, global_ex_scaler)


def load_kddcup2022_sdwpf(data_root: str,
                          frequency: Literal['10min', '30min', '1hour', '6hour', '12hour', '1day'],
                          ds_params: dict,
                          power_factor: float = 1.,
                          split_ratio: float = 0.8,
                          use_ex_ts: bool = False,
                          scaler: Scale = Scale(),
                          ex_scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load KDD Cup 2022 SDWPF datasets for model training and evaluation.

        The data fields are: ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv'].
        The time interval is **10 min**.

        :param data_root: the root directory of the whole datasets.
        :param frequency: the frequency of the time series, ['10min', '30min', '1hour', '6hour', '12hour', '1day'].
        :param ds_params: the dataset parameters.
        :param power_factor: the power factor, default is 1.
        :param split_ratio: the ratio of training set. Split according to csv files (or turbines).
        :param use_ex_ts: whether to use time as exogenous factor, default is False.
        :param scaler: the global scaler class for the target datasets, default is Scale().
        :param ex_scaler: the global scaler class for the exogenous datasets, default is Scale().
        :return:
    """
    data_root = data_root + r'time_series/energy_wind/SDWPF/4_multiple_uts'

    data = get_kddcup2022_sdwpf_list(data_root, frequency, power_factor)
    data = list(zip(*data))
    random.shuffle(data)
    split_pos = int(len(data) * split_ratio)  # file numbers

    train_data, val_data = data[:split_pos], data[split_pos:]
    train_time_list, train_ts_list, train_ex_list = list(zip(*train_data))
    val_time_list, val_ts_list, val_ex_list = list(zip(*val_data))

    train_ts_list = [torch.tensor(arr) for arr in train_ts_list]
    val_ts_list = [torch.tensor(arr) for arr in val_ts_list]
    train_ex_list = [torch.tensor(arr) for arr in train_ex_list]
    val_ex_list = [torch.tensor(arr) for arr in val_ex_list]

    train_ex_ts, val_ex_ts = None, None
    global_scaler, global_ex_scaler = time_series_scaler(train_ts_list, scaler), None
    if use_ex_ts:
        train_ex_ts, val_ex_ts = train_ex_list, val_ex_list
        global_ex_scaler = time_series_scaler(train_ex_ts, ex_scaler)

    ds_params.update({'split_ratio': 1., 'split': 'train'})
    train_ds = UTSDataset(train_ts_list, ex_ts=train_ex_ts, **ds_params, stride=1)
    val_ds = UTSDataset(val_ts_list, ex_ts=val_ex_ts, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_scaler, global_ex_scaler)


"""
    Simulation datasets.
"""


def load_grid_forming_converter(data_root: str,
                                ds_params: dict,
                                split_ratio: float = 0.8,
                                use_features: bool = False,
                                scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load grid forming converter datasets for training and evaluation.

        Data fields are:
            ['simTime',
            'Vgq', 'Vgd', 'Pinertia', 'Pdamping', 'SCR', 'XbyR', 'IsOvercurrent', 'GridMag', 'GridFreq', 'GridPhase',
            'Pref_real', 'Qref_real', 'VolRef', 'Qload', 'Igd', 'Igq']

        In the perspective of the grid forming converter, the inputs fields are:
            ['SCR', 'XbyR', 'GridMag', 'GridFreq', 'GridPhase', 'Pref_real', 'Qref_real', 'VolRef', 'Qload']
        The output fields are the remaining fields:
            ['Vgq', 'Vgd', 'Pinertia', 'Pdamping', 'Igd', 'Igq', 'IsOvercurrent']

        :param data_root: the root directory of the whole datasets.
        :param ds_params: the ``UTSDataset`` parameters.
        :param split_ratio: the ratio of training set. Split according to csv files (simulation times).
        :param use_features: whether to use features in the datasets.
        :param scaler: the global scaler for the datasets, default is ``Scale()``.
        :return:
    """

    def get_gfm_data(filenames: list[str], fields: list[str],
                     input_fields: list[str], output_fields: list[str],
                     tqdm_desc: str = 'Loading') -> tuple[list, list]:
        inputs_list, outputs_list = [], []
        with tqdm(filenames, leave=False, file=sys.stdout) as pbar:
            pbar.set_description(tqdm_desc)
            for name in filenames:
                df = pd.read_csv(os.path.join(grid_forming_data_root, name), header=None, names=fields)
                inputs_list.append(df[input_fields].values.astype(np.float32))
                outputs_list.append(df[output_fields].values.astype(np.float32))
                pbar.update(1)
        return inputs_list, outputs_list

    grid_forming_data_root = os.path.join(data_root, 'time_series/simulation_gfm/gfm_sim_disturbance/')
    csv_filenames = [name for name in sorted(os.listdir(grid_forming_data_root)) if name.endswith('.csv')]

    columns = ['simTime', 'Vgq', 'Vgd', 'Pinertia', 'Pdamping', 'SCR', 'XbyR', 'IsOvercurrent',
               'GridMag', 'GridFreq', 'GridPhase', 'Pref_real', 'Qref_real', 'VolRef', 'Qload', 'Igd', 'Igq']
    input_columns = ['SCR', 'XbyR', 'GridMag', 'GridFreq', 'GridPhase', 'Pref_real', 'Qref_real', 'VolRef', 'Qload']

    # output_columns = ['Vgq', 'Vgd', 'Pinertia', 'Pdamping', 'Igd', 'Igq', 'IsOvercurrent']
    output_columns = ['Vgq', 'Vgd', 'Pinertia', 'Pdamping', 'Igd', 'Igq']
    # output_columns = ['Vgq', 'Vgd', 'Igd', 'Igq']
    # output_columns = ['Vgq', 'Vgd']  # common units

    random.shuffle(csv_filenames)
    split_position = int(len(csv_filenames) * split_ratio)

    train_filenames, val_filenames = csv_filenames[:split_position], csv_filenames[split_position:]
    train_feature_list, train_ts_list = \
        get_gfm_data(train_filenames, columns, input_columns, output_columns, 'Loading train')
    val_feature_list, val_ts_list = get_gfm_data(val_filenames, columns, input_columns, output_columns, 'Loading val')

    train_ts_list = [torch.tensor(arr) for arr in train_ts_list]
    train_feature_list = [torch.tensor(arr) for arr in train_feature_list]
    val_ts_list = [torch.tensor(arr) for arr in val_ts_list]
    val_feature_list = [torch.tensor(arr) for arr in val_feature_list]

    # add noise

    train_ex_ts, val_ex_ts = None, None
    global_soh_scaler, global_feature_scaler = time_series_scaler(train_ts_list, scaler), None
    if use_features:
        train_ex_ts, val_ex_ts = train_feature_list, val_feature_list
        global_feature_scaler = time_series_scaler(train_ex_ts, scaler)

    ds_params.update({'split_ratio': 1., 'split': 'train'})
    train_ds = UTSDataset(train_ts_list, ex_ts=train_ex_ts, **ds_params, stride=1)
    val_ds = UTSDataset(val_ts_list, ex_ts=val_ex_ts, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_soh_scaler, global_feature_scaler)
