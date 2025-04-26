#!/usr/bin/env python
# encoding: utf-8

"""

    Prepare commonly used time series forecasting datasets.

    (1) ETT datasets: ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv.
        Exchange rate: exchange_rate.csv.
        Jena Climate: mpi_roof_2010a.csv, mpi_saale_2010a.csv.
        UCI Electricity: electricity_20110101_20150101.csv, electricity_20160701_20190702.csv.
        US PEMS 03 04 07 08: pems03_flow.csv, pems04_flow.csv, pems07_flow.csv, pems08_flow.csv.
        US PEMS Traffic: traffic_20160701_20180702.csv.
        US CDC Flu Activation Level: US_regional_flu_level_20181004_20250322.csv.
        US CDC ILI: US_National_ILI_1997_2025.csv.
        WHO Japan ILI: Japan_ILI_19961006_20250309.csv.

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

dataset_meta_lookup = {
    'ETTh1': {
        'path': '{root}/Github_ETT_small/ETTh1.csv',
        'columns': {
            'univariate': ['OT'],
            'multivariate': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
            'exogenous': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
            'time': 'Date'
        }
    },
    'ETTh2': {
        'path': '{root}/Github_ETT_small/ETTh2.csv',
        'columns': {
            'univariate': ['OT'],
            'multivariate': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
            'exogenous': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        }
    },
    'ETTm1': {
        'path': '{root}/Github_ETT_small/ETTm1.csv',
        'columns': {
            'univariate': ['OT'],
            'multivariate': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
            'exogenous': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        }
    },
    'ETTm2': {
        'path': '{root}/Github_ETT_small/ETTm2.csv',
        'columns': {
            'univariate': ['OT'],
            'multivariate': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
            'exogenous': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        }
    },
    'ExchangeRate': {
        'path': '{root}/Github_exchange_rate/exchange_rate.csv',
        'columns': {
            'univariate': ['Singapore'],
            'multivariate': ['Australia', 'British', 'Canada', 'Switzerland', 'China', 'Japan', 'New Zealand',
                             'Singapore']
        }
    },
    'JenaClimate': {
        'path': '{root}/MaxPlanck_Jena_Climate/mpi_roof_2010a.csv',
        'columns': {
            'univariate': ['T (degC)'],
            'multivariate': ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)',
                             'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)',
                             'max. wv (m/s)', 'wd (deg)', 'rain (mm)', 'raining (s)', 'SWDR (W/m)', 'PAR (mol/m/s)',
                             'max. PAR (mol/m/s)', 'Tlog (degC)', 'CO2 (ppm)']
        }
    },
    'Electricity': {
        'path': '{root}/UCI_Electricity/electricity_20160701_20190702.csv',
        'columns': {
            'univariate': ['320'],
            'multivariate': ['{}'.format(i) for i in range(320 + 1)]
        }
    },
    'PeMS03': {
        'path': '{root}/US_PEMS_03_04_07_08/pems03_flow.csv',
        'columns': {
            'univariate': ['357'],
            'multivariate': [str(i) for i in range(358)]
        }
    },
    'PeMS04': {
        'path': '{root}/US_PEMS_03_04_07_08/pems04_flow.csv',
        'columns': {
            'univariate': ['306'],
            'multivariate': [str(i) for i in range(307)]
        }
    },
    'PeMS07': {
        'path': '{root}/US_PEMS_03_04_07_08/pems07_flow.csv',
        'columns': {
            'univariate': ['882'],
            'multivariate': [str(i) for i in range(883)]
        }
    },
    'PeMS08': {
        'path': '{root}/US_PEMS_03_04_07_08/pems08_flow.csv',
        'columns': {
            'univariate': ['169'],
            'multivariate': [str(i) for i in range(170)]
        }
    },
    'Traffic': {
        'path': '{root}/US_PEMS_Traffic/traffic_20160701_20180702.csv',
        'columns': {
            'univariate': ['861'],
            'multivariate': [str(i) for i in range(861 + 1)]
        }
    },
    'US_CDC_Flu': {
        'path': '{root}/US_CDC_Flu_Activation_Level/US_regional_flu_level_20181004_20250322.csv',
        'columns': {
            'univariate': ['Commonwealth of the Northern Mariana Islands'],
            'multivariate': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
                             'Delaware', 'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois',
                             'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
                             'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
                             'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
                             'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
                             'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
                             'West Virginia', 'Wisconsin', 'Wyoming', 'New York City', 'Puerto Rico', 'Virgin Islands',
                             'Commonwealth of the Northern Mariana Islands']
        }
    },
    'US_CDC_ILI': {
        'path': '{root}/US_CDC_ILI/US_National_ILI_1997_2025.csv',
        'columns': {
            'univariate': ['ILITOTAL'],
            'multivariate': [['WEIGHTED ILI', 'UNWEIGHTED ILI', 'AGE 0-4',
                              # 'AGE 25-49', # TODO: 数据存在字母X
                              # 'AGE 25-64',  # TODO: 数据存在字母X
                              'AGE 5-24',
                              # 'AGE 50-64',   # TODO: 数据存在字母XX
                              'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS']]
        }
    },
    'WHO_JAPAN_ILI': {
        'path': '{root}/WHO_Japan_ILI/Japan_ILI_19961006_20250309.csv',
        'columns': {
            'univariate': ['ILI_ACTIVITY'],
            'multivariate': ['AH1', 'AH1N12009', 'AH3', 'AH5', 'ANOTSUBTYPED', 'INF_A', 'BVIC', 'BYAM',
                             'BNOTDETERMINED', 'INF_B', 'INF_ALL', 'INF_NEGATIVE', 'ILI_ACTIVITY']
        }
    },
}


def load_dataset_sst(mts_data_root: str, ds_name: str,
                     task: Literal['univariate', 'multivariate'] = 'univariate',
                     use_ex_vars: bool = False, time_as_feature: TimeAsFeature = None,
                     input_window_size: int = 96, output_window_size: int = 24, horizon: int = 1, stride: int = 1,
                     train_ratio: float = 0.8, val_ratio: float = None,
                     scaler: Scale = None, ex_scaler: Scale = None):
    """
    Unified function to load datasets based on configuration.

    :param mts_data_root: The root directory where the dataset files are stored.
    :param ds_name: The key identifying the dataset in the `DATASET_CONFIG` dictionary.
                        Must be one of ["ETT", "ExchangeRate", "JenaClimate", "Electricity", "US_PEMS",
                        "US_PEMS_Traffic", "US_CDC_FLU", "US_CDC_ILI", "WHO_JAPAN_ILI"].
    :param task: The type of task to perform. Can be 'univariate' or 'multivariate'. Default is 'univariate'.
    :param use_ex_vars: Whether to use exogenous variables. Default is False.
    :param time_as_feature: An instance of `TimeAsFeature` to extract time-based features from the dataset.
    :param input_window_size: The size of the input window (number of time steps). Default is 96.
    :param output_window_size: The size of the output window (number of time steps). Default is 24.
    :param horizon: The number of time steps between the input and output windows. Default is 1.
    :param stride: The number of time steps between consecutive windows. Default is 1.
    :param train_ratio: The ratio of the dataset to use for training. Must be in the range (0, 1]. Default is 0.8.
    :param val_ratio: The ratio of the dataset to use for validation. Must be in the range (0, 1). Default is None.
    :param scaler: An instance of `Scale` to normalize the target variables, a global scaler. Default is None.
    :param ex_scaler: An instance of `Scale` to normalize the exogenous variables, a global scaler. Default is None.

    :return: A tuple containing:
             - (train_dataset, val_dataset, test_dataset): The datasets split into training, validation, and testing sets.
             - (scaler, ex_scaler): The scalers used for normalizing the target and exogenous variables.
    """
    assert ds_name in dataset_meta_lookup, f'Invalid dataset name: {ds_name}. Must be one of {list(dataset_meta_lookup.keys())}.'
    assert task in ('univariate', 'multivariate'), f'Invalid task: {task}.'

    assert input_window_size > 0, 'Invalid input window size: {}'.format(input_window_size)
    assert output_window_size > 0, 'Invalid output window size: {}'.format(output_window_size)
    assert stride > 0, 'Invalid stride: {}'.format(stride)

    assert 0 < train_ratio <= 1, f'Invalid train_ratio: {train_ratio}. It must be in the range (0, 1].'
    if val_ratio is not None:
        assert 0 < val_ratio < 1, f'Invalid val_ratio: {val_ratio}. It must be in the range (0, 1).'
        assert 0 < train_ratio + val_ratio < 1, \
            f'Invalid test_ratio: {train_ratio} + {val_ratio}. It must be in the range (0, 1).'

    meta_info = dataset_meta_lookup[ds_name]
    csv_filename = meta_info['path'].format(root=mts_data_root)

    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f'File not found: {csv_filename}')

    print(f'Loading: {csv_filename}')
    df = pd.read_csv(csv_filename)

    sst_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    vars = meta_info['columns'][task]
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    if scaler is not None and type(scaler) != type(Scale()):
        scaler = scaler.fit(target_tensor)

    if use_ex_vars:
        ex_vars = meta_info['columns']['exogenous']
        ex_df = df.loc[:, ex_vars]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)
        sst_params['ex_ts'] = ex_tensor

        if ex_scaler is not None and type(ex_scaler) != type(Scale()):
            ex_scaler = ex_scaler.fit(ex_tensor)

    if time_as_feature is not None:
        time_var = meta_info['columns']['time']
        df[time_var] = pd.to_datetime(df[time_var])
        time_features = time_as_feature(df[time_var].dt)
        time_feature_tensor = torch.tensor(time_features)
        sst_params["ex_ts2"] = time_feature_tensor

    if train_ratio == 1.0:
        train_ds = SSTDataset(**sst_params)
        return (train_ds, None, None), (scaler, ex_scaler)

    if val_ratio is None or val_ratio == 0:
        train_ds = SSTDataset(**sst_params, stride=stride).split(train_ratio, 'train', 'train')
        val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size']).split(train_ratio, 'val', 'val')
        return (train_ds, val_ds, None), (scaler, ex_scaler)

    ratio1 = train_ratio + val_ratio        # split ratio of (train + val) / test
    ratio2 = train_ratio / ratio1           # split ratio of train / (train + val)
    o_stride = sst_params['output_window_size']
    train_ds = SSTDataset(**sst_params, stride=stride).split(ratio1, 'train').split(ratio2, 'train', 'train')
    val_ds = SSTDataset(**sst_params, stride=o_stride).split(ratio1, 'train').split(ratio2, 'val', 'val')
    test_ds = SSTDataset(**sst_params, stride=o_stride).split(ratio1, 'val', 'test')

    return (train_ds, val_ds, test_ds), (scaler, ex_scaler)
