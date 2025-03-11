#!/usr/bin/env python
# encoding: utf-8

"""
    Read persisted dataset to memory, maybe dataframes or numpy arrays.
"""
import os, random

import numpy as np
import pandas as pd
from typing import Literal

"""
    Disease datasets.
"""


def get_xmcdc(data_root: str = '../dataset/xmcdc/',
              frequency: Literal['daily', 'weekly'] = 'daily',
              disease_names: list[str] = ['手足口病', '肝炎', '其他感染性腹泻']):
    """
        Outpatient cases provided by Xiamen Center for Disease Prevention and Control.
        :param data_root: the data root directory.
        :param frequency: [daily | hourly]
        :param disease_names: select from ['手足口病', '肝炎', '其他感染性腹泻'].
        :return: numpy arrays.
    """
    csv_file = data_root + r'{}_outpatients_2011_2020.csv'.format(frequency)

    df = pd.read_csv(csv_file)

    disease_df = df[disease_names]  # target array
    disease_array = np.array(disease_df, dtype=np.float32)

    weather_df = df.iloc[:, 1:17]
    weather_array = np.array(weather_df, dtype=np.float32)

    bsi_df = df.iloc[:, 17:26]
    bsi_array = np.array(bsi_df, dtype=np.float32)

    time_array = df['Date'].values
    if frequency == 'daily':
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month.values
        df['day'] = df['Date'].dt.day.values
        df['dayofweek'] = df['Date'].dt.dayofweek.values
        df['timestamp'] = (df['Date'] - df['Date'].min()).dt.days.values

        time_array = df[['timestamp', 'day', 'dayofweek']].values.astype(np.float32)

    # fine-grained data: need to be added

    return time_array, disease_array, weather_array, bsi_array



"""
    Energy - wind datasets.
"""


def get_greek_wind(data_root: str,
                   frequency: Literal['1hour', '6hour', '1day', '12hour'],
                   factor: float = 1.) -> tuple[list, list, list]:
    """
        SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022
        :param data_root: the multiple uts greek wind power time series root directory.
        :param frequency: [1day | 12hour | 6hour | 1hour]
        :param factor: the factor to scale the wind power values.
        :return: numpy arrays (time, target, exogenous).
    """
    data_root = data_root + '/' + frequency

    time_array_list, wind_power_array_list, exogenous_array_list = [], [], []
    for csv in sorted(os.listdir(data_root)):
        if csv.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_root, csv), index_col=None)

            df['time'] = pd.to_datetime(df['time'])

            time_array = df['time']
            wind_power_array = df['power(MW)'].values.reshape(-1, 1).astype(np.float32) * factor
            exogenous_array = df.iloc[:, 2:].values.astype(np.float32)

            time_array_list.append(time_array)
            wind_power_array_list.append(wind_power_array)
            exogenous_array_list.append(exogenous_array)

    return time_array_list, wind_power_array_list, exogenous_array_list


def get_kddcup2022_sdwpf(wpf_data_root: str = '~/data/time_series/energy_wind/SDWPF/2_mts_csv',
                         frequency: Literal['10min', '30min', '1hour', '6hour', '12hour', '1day'] = '10min'):
    """
        SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022
        :param wpf_data_root: the WPF time series root directory.
        :param frequency: [10min | 30min | 1hour | 6hour | 12hour | 1day]
        :return: numpy arrays.
    """
    csv_file = wpf_data_root + '/SDWPF_Patv_cubicspline_{}.csv'.format(frequency)

    df = pd.read_csv(csv_file)
    date_df = df.iloc[:, 0:1]

    patv_df = df.iloc[:, 1:]
    patv_array = np.array(patv_df, dtype=np.float64) / 1000.0  # convert to MW

    return date_df, patv_array


def get_kddcup2022_sdwpf_list(data_root: str,
                              frequency: Literal['10min', '30min', '1hour', '6hour', '12hour', '1day'] = '10min',
                              factor: float = 1.) -> tuple[list, list, list]:
    """
        SDWPF: A Dataset for Spatial Dynamic Wind Power Forecasting Challenge at KDD Cup 2022
        :param data_root: the WPF time series root directory.
        :param frequency: [10min | 30min | 1hour | 6hour | 12hour | 1day]
        :return: numpy arrays.
    """
    data_root = data_root + '/' + frequency

    time_array_list, wind_power_array_list, exogenous_array_list = [], [], []
    for csv in sorted(os.listdir(data_root)):
        if csv.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_root, csv), index_col=None)

            df['Datetime'] = pd.to_datetime(df['Datetime'])

            time_array = df['Datetime']
            wind_power_array = df['Patv'].values.reshape(-1, 1).astype(np.float32) * factor
            exogenous_array = df.iloc[:, 1:-1].values.astype(np.float32)

            time_array_list.append(time_array)
            wind_power_array_list.append(wind_power_array)
            exogenous_array_list.append(exogenous_array)

    return time_array_list, wind_power_array_list, exogenous_array_list


"""
    Energy - battery datasets.
"""


def get_nenergy_2019_battery(csv_filenames: list[str],
                             target: Literal['RUL(cycle)', 'PCL', 'SoH'] or str = 'RUL(cycle)',
                             use_smooth: bool = False) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
        The features used in tiv2023_deephpm are:
            ['cycle_id', 'slope', 'intercept', 'Tavg', 'IR', 'chargetime',
            'dQdV_max', 'dQdV_min', 'dQdV_var', 'RUL(cycle)', 'PCL']

        K.A. Severson, P.M. Attia, et al., nature energy, 2019.
        'Data driven prediction of battery cycle life before capacity degradation',
        URL: https://www.nature.com/articles/s41560-019-0356-8

        Download: https://data.matr.io/1/

        :param csv_filenames: the csv filenames of the battery cycle life datasets.
        :param target: the prediction target, the values maybe one of 'RUL(cycle)', 'PCL' or 'SoH'.
        :param use_smooth: weather to use interpolate or not.
        :return: a tuple of two lists,
                 the first list is the SoH time series,
                 the second list is the feature time series.
    """

    df_list = []
    for name in sorted(csv_filenames):
        df = pd.read_csv(name)
        df['SoH'] = 1 - df['PCL']

        if use_smooth:
            columns = ['slope', 'intercept', 'Tavg', 'IR', 'chargetime', 'dQdV_max', 'dQdV_min', 'dQdV_var']
            df[columns] = df[columns].rolling(window=10, min_periods=1).mean().round(5)
            df[target] = df[target].round(5)

        df_list.append(df)

    target_ts_list, feature_ts_list = [], []

    columns = ['cycle_id', 'slope', 'intercept', 'Tavg', 'IR', 'chargetime', 'dQdV_max', 'dQdV_min', 'dQdV_var']

    for df in df_list:
        target_ts = df[target].values.reshape(-1, 1).astype(np.float32)
        # target_ts = target_ts * (100 if target == 'SoH' else 1)
        feature_ts = df[columns].values.astype(np.float32)

        target_ts_list.append(target_ts)
        feature_ts_list.append(feature_ts)

    return target_ts_list, feature_ts_list


def get_nacom_2024_battery(battery_data_root: str,
                           cathode_material: Literal['NMC2.1', 'LMO', 'NMC21', 'LFP'] = 'NMC2.1',
                           soc_condition: list[float] or str = 'all'):
    """
        :param battery_data_root: the root directory of the nature communication 2024 battery data.
        :param cathode_material: the battery material, one of ['NMC2.1','NMC21','LMO','LFP'].
        :param soc_condition: the SOC condition, one of ['all', 0.1, 0.2, 0.3, 0.4, 0.5].
        :return: soc_condition, condition, filtered_data: numpy arrays.
    """
    material_to_filename = {
        'NMC2.1': 'NMC_2.1Ah_W_3000.xlsx',
        'NMC21': 'NMC_21Ah_W_3000.xlsx',
        'LMO': 'LMO_10Ah_W_3000.xlsx',
        'LFP': 'LFP_35Ah_W_3000.xlsx'
    }

    filename = os.path.join(battery_data_root, material_to_filename.get(cathode_material))
    df = pd.read_excel(filename, sheet_name="SOC ALL", engine='openpyxl')

    df['SOC'] /= 100  # Normalize SOC by dividing by 100
    # Filter data to include only rows where SOC is less than or equal to 50% (0.5 after normalization)
    filtered_df = df[df['SOC'] <= 0.5]

    if soc_condition != 'all':
        filtered_df = df[df['SOC'] in soc_condition].values.astype(np.float32)

    feature_array = filtered_df.loc[:, 'U1':'U21'].values.astype(np.float32)

    soc_array = filtered_df['SOC'].values
    soh_array = filtered_df['SOH'].values
    condition_array = np.column_stack((soc_array, soh_array))

    return feature_array, condition_array


def get_dfdq_battery(battery_data_root: str,
                     train_or_test: Literal['train', 'test1'] = 'train') -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
        Data fields are:
        ['datetime1', 'datetime2', '循环号', '充电时长ms', '放电时长ms', '充放电容量Ah']
        :param battery_data_root: the root directory of the battery time series in csv files.
        :param train_or_test: ['train' | 'test1']
    """
    csv_file_dir = os.path.join(battery_data_root, '2_uts/{}'.format(train_or_test))

    csv_filenames = [filename for filename in os.listdir(csv_file_dir) if filename.endswith('.csv')]

    df_list = []
    for filename in sorted(csv_filenames):
        csv_full_name = os.path.join(csv_file_dir, filename)
        df = pd.read_csv(csv_full_name)
        df_list.append(df)

    capacity_list, feature_list = [], []
    feature_columns = ['循环号', '充电时长ms', '放电时长ms']
    for df in df_list:
        capacity_ts = df['充放电容量Ah'].values.reshape(-1, 1)
        feature_ts = df[feature_columns].values

        capacity_list.append(capacity_ts)
        feature_list.append(feature_ts)

    return capacity_list, feature_list
