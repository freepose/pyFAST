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

"""

    ETT dataset.

"""


def get_ett_columns(return_vars: Literal['univariate', 'multivariate', 'exogenous', 'all'] = 'univariate') -> List:
    """
        Get the columns of ETT dataset.

        :param return_vars: The type of variables to return. ``univariate``, ``multivariate``, ``exogenous``, and ``all``.
    """
    columns = ['Date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

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
                 input_window_size: int = 96, output_window_size: int = 24, horizon: int = 1, stride: int = 1,
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

    csv_file = '{}/{}.csv'.format(ett_data_root, subset)

    if not os.path.exists(csv_file):
        raise FileNotFoundError("File not found: {}".format(csv_file))

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
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
    # target_tensor = MinMaxScale().fit_transform(target_tensor)  # Previous software were worked with this
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


"""

    Exchange rate dataset.

"""


def get_exchange_rate_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate'):
    """
        Get the columns of exchange rate dataset.

        :return: The columns of exchange rate dataset.
    """

    columns = ['Date', 'Australia', 'British', 'Canada', 'Switzerland', 'China', 'Japan', 'New Zealand', 'Singapore']

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_exchange_rate_sst(exchange_rate_data_root: str,
                           vars: List[str] = None,
                           time_as_feature: TimeAsFeature = None,
                           split_ratio: float = 0.8,
                           input_window_size: int = 96, output_window_size: int = 1, horizon: int = 1, stride: int = 1,
                           scaler: Scale = None):
    """
        Load exchange rate dataset as ``SSTDataset``. The train / val split type is **intra** time series.

        The frequency of exchange rate dataset is daily.

        :param exchange_rate_data_root: The root directory of exchange rate dataset. For example: ``~/data/time_series/general_mts/exchange_rate``.
        :param vars: The target variable(s) of exchange rate dataset. Default is None, and the ``['Singapore']`` is chosen as target.
        :param time_as_feature: The time feature class. Default is ``None``.
        :param split_ratio: The split ratio of train and test set. Default is ``0.8``. If 1, the whole dataset will be used as train set.
        :param input_window_size: The input window size. Default is ``24``.
        :param output_window_size: The output window size. Default is ``1``.
        :param horizon: The time steps between input window and output window. Default is ``1``.
        :param stride: The time steps between two consecutive (input / output) windows. Default is ``1``.
        :param scaler: The scaler for the target variable(s). Default is ``None``, which means no scaling.
        :return:
    """
    assert 0 < split_ratio <= 1, "Invalid split ratio: {}".format(split_ratio)
    assert input_window_size > 0, "Invalid input window size: {}".format(input_window_size)
    assert output_window_size > 0, "Invalid output window size: {}".format(output_window_size)
    assert stride > 0, "Invalid stride: {}".format(stride)

    csv_file = '{}/exchange_rate.csv'.format(exchange_rate_data_root)

    if not os.path.exists(csv_file):
        raise FileNotFoundError("File not found: {}".format(csv_file))

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    if vars is None:
        vars = ['Singapore']

    target_df = df.loc[:, vars]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    # target_tensor = MinMaxScale().fit_transform(target_tensor)  # Previous software were worked with this
    sst_params['ts'] = target_tensor

    if scaler is not None and type(scaler) != type(Scale()):
        scaler = scaler.fit(target_tensor)

    if time_as_feature is not None:
        df['Date'] = pd.to_datetime(df['Date'])
        time_dt = df['Date'].dt
        time_feature_array = time_as_feature(time_dt)
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    MaxPlanck Jena Climate dataset.

"""


def get_jena_climate_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate'):
    """
        Get the columns of Jena Climate dataset.

        :return: The columns of Jena Climate dataset.
    """
    columns = ['Date', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)',
               'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)',
               'rain (mm)', 'raining (s)', 'SWDR (W/m)', 'PAR (mol/m/s)', 'max. PAR (mol/m/s)', 'Tlog (degC)',
               'CO2 (ppm)']

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_jena_climate_sst(jena_climate_data_root: str,
                          vars: List[str] = None,
                          time_as_feature: TimeAsFeature = None,
                          split_ratio: float = 0.8,
                          input_window_size: int = 96,
                          output_window_size: int = 24,
                          horizon: int = 1,
                          stride: int = 1,
                          scaler: Scale = None):
    """
    Load the Jena Climate dataset and return it in SSTDataset format.

    :param jena_climate_data_root: Root directory of the Jena Climate dataset.
    :param vars: List of target variables. Defaults to None, selecting all variables.
    :param time_as_feature: Whether to include time as a feature. Defaults to False.
    :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
    :param input_window_size: Size of the input window. Defaults to 96.
    :param output_window_size: Size of the output window. Defaults to 24.
    :param horizon: Time steps between the input and output windows. Defaults to 1.
    :param stride: Time steps between consecutive windows. Defaults to 1.
    :param scaler: Scaler for the target variables. Defaults to None.
    :return: (training dataset, validation dataset), scaler
    """
    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(jena_climate_data_root, 'mpi_roof_2010a.csv')

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = ['CO2 (ppm)']  # Default to all variables except the first (time column)

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    UCI Electricity dataset.

"""


def get_uci_electricity_columns(return_vars: Literal['univariate', 'multivariate'] = 'univariate'):
    """
        Get the columns of UCI Electricity dataset.

        :return: The columns of UCI Electricity dataset.
    """
    columns = ['Date'] + ['{}'.format(i) for i in range(320 + 1)]

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_uci_electricity_sst(uci_electricity_data_root: str,
                             vars: List[str] = None,
                             time_as_feature: TimeAsFeature = None,
                             split_ratio: float = 0.8,
                             input_window_size: int = 96,
                             output_window_size: int = 96,
                             horizon: int = 1,
                             stride: int = 1,
                             scaler: Scale = None):
    """
    Load the UCI Electricity dataset and return it in SSTDataset format.

    :param uci_electricity_data_root: Root directory of the UCI Electricity dataset.
    :param location: Location of the dataset, either 'roof' or 'saale'. Defaults to 'roof'.
    :param vars: List of target variables. Defaults to None, selecting all variables.
    :param time_as_feature: Whether to include time as a feature. Defaults to False.
    :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
    :param input_window_size: Size of the input window. Defaults to 96.
    :param output_window_size: Size of the output window. Defaults to 24.
    :param horizon: Time steps between the input and output windows. Defaults to 1.
    :param stride: Time steps between consecutive windows. Defaults to 1.
    :param scaler: Scaler for the target variables. Defaults to None.
    :return: (training dataset, validation dataset), scaler
    """
    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(uci_electricity_data_root, 'electricity_{}.csv'.format('20160701_20190702'))

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = [df.columns[-1]]

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    US PEMS 03 04 07 08 dataset.

"""


def get_us_pems_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate',
                        subset: Literal['pems03', 'pems04', 'pems07', 'pems08'] = 'pems03'):
    """
        Get the columns of US PEMS dataset.

        :return: The columns of US PEMS dataset.
    """
    assert subset in ['pems03', 'pems04', 'pems07', 'pems08'], "Invalid subset: {}".format(subset)

    if subset == 'pems03':
        columns = columns = ['Date'] + [str(i) for i in range(357 + 1)]
    elif subset == 'pems04':
        columns = ['Date'] + [str(i) for i in range(306 + 1)]
    elif subset == 'pems07':
        columns = ['Date'] + [str(i) for i in range(882 + 1)]
    elif subset == 'pems08':
        columns = ['Date'] + [str(i) for i in range(169 + 1)]
    else:
        return None

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_us_pems_sst(us_pems_data_root: str,
                     subset: Literal['pems03', 'pems04', 'pems07', 'pems08'] = 'pems03',
                     vars: List[str] = None,
                     time_as_feature: TimeAsFeature = None,
                     split_ratio: float = 0.8,
                     input_window_size: int = 96,
                     output_window_size: int = 96,
                     horizon: int = 1,
                     stride: int = 1,
                     scaler: Scale = None):
    """
    Load the US PEMS 03 04 07 08 dataset and return it in SSTDataset format.

    :param us_pems_data_root: Root directory of the US PEMS 03 04 07 08 dataset.
    :param subset: Subset of the dataset, ``pems03``, ``pems04``, ``pems07``, and ``pems08``.
    :param vars: List of target variables. Defaults to None, selecting all variables.
    :param time_as_feature: Whether to include time as a feature. Defaults to False.
    :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
    :param input_window_size: Size of the input window. Defaults to 96.
    :param output_window_size: Size of the output window. Defaults to 96.
    :param horizon: Time steps between the input and output windows. Defaults to 1.
    :param stride: Time steps between consecutive windows. Defaults to 1.
    :param scaler: Scaler for the target variables. Defaults to None.
    :return: (training dataset, validation dataset), scaler
    """
    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(us_pems_data_root, '{}_flow.csv'.format(subset))

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = [df.columns[-1]]

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    US PEMS Traffic dataset.

"""


def get_us_pems_traffic_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate'):
    """
        Get the columns of Traffic dataset.

        :return: The columns of Traffic dataset.
    """
    columns = ['Date'] + [str(i) for i in range(861 + 1)]

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_us_pems_traffic_sst(us_pems_traffic_data_root: str,
                             vars: List[str] = None,
                             time_as_feature: TimeAsFeature = None,
                             split_ratio: float = 0.8,
                             input_window_size: int = 96,
                             output_window_size: int = 96,
                             horizon: int = 1,
                             stride: int = 1,
                             scaler: Scale = None):
    """
    Load the US PEMS Traffic dataset and return it in SSTDataset format.

    :param us_pems_traffic_data_root: Root directory of the US PEMS Traffic dataset.
    :param vars: List of target variables. Defaults to None, selecting all variables.
    :param time_as_feature: Whether to include time as a feature. Defaults to False.
    :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
    :param input_window_size: Size of the input window. Defaults to 96.
    :param output_window_size: Size of the output window. Defaults to 96.
    :param horizon: Time steps between the input and output windows. Defaults to 1.
    :param stride: Time steps between consecutive windows. Defaults to 1.
    :param scaler: Scaler for the target variables. Defaults to None.
    :return: (training dataset, validation dataset), scaler
    """
    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(us_pems_traffic_data_root, 'traffic_20160701_20180702.csv')

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = [df.columns[-1]]

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    US CDC Flu Activation Level dataset.

"""


def get_us_cdc_flu_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate'):
    """
        Get the columns of US CDC Flu Activation Level dataset.

        :return: The columns of US CDC Flu Activation Level dataset.
    """
    columns = ['Date', 'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware',
               'District of Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas',
               'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
               'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
               'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
               'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
               'West Virginia', 'Wisconsin', 'Wyoming', 'New York City', 'Puerto Rico', 'Virgin Islands',
               'Commonwealth of the Northern Mariana Islands']

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_us_cdc_flu_activation_level_sst(us_cdc_flu_activation_level_data_root: str,
                                         vars: List[str] = None,
                                         time_as_feature: TimeAsFeature = None,
                                         split_ratio: float = 0.8,
                                         input_window_size: int = 96,
                                         output_window_size: int = 96,
                                         horizon: int = 1,
                                         stride: int = 1,
                                         scaler: Scale = None):
    """
    Load the US CDC Flu Activation Level dataset and return it in SSTDataset format.

    :param us_cdc_flu_activation_level_data_root: Root directory of the US CDC Flu Activation Level dataset.
    :param vars: List of target variables. Defaults to None, selecting all variables.
    :param time_as_feature: Whether to include time as a feature. Defaults to False.
    :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
    :param input_window_size: Size of the input window. Defaults to 96.
    :param output_window_size: Size of the output window. Defaults to 96.
    :param horizon: Time steps between the input and output windows. Defaults to 1.
    :param stride: Time steps between consecutive windows. Defaults to 1.
    :param scaler: Scaler for the target variables. Defaults to None.
    :return: (training dataset, validation dataset), scaler
    """
    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(us_cdc_flu_activation_level_data_root, 'US_regional_flu_level_20181004_20250322.csv')

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = [df.columns[-1]]

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    US CDC ILI dataset.

"""


def get_us_cdc_ili_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate'):
    """
        Get the columns of US CDC ILI dataset.

        :return: The columns of US CDC ILI dataset.
    """
    columns = ['Date', 'WEIGHTED ILI', 'UNWEIGHTED ILI', 'AGE 0-4',
               # 'AGE 25-49', # TODO: 数据存在字母X
               # 'AGE 25-64',  # TODO: 数据存在字母X
               'AGE 5-24',
               # 'AGE 50-64',   # TODO: 数据存在字母XX
               'AGE 65', 'ILITOTAL', 'NUM. OF PROVIDERS', 'TOTAL PATIENTS']

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_us_cdc_ili_sst(us_cdc_ili_data_root: str,
                        vars: List[str] = None,
                        time_as_feature: TimeAsFeature = None,
                        split_ratio: float = 0.8,
                        input_window_size: int = 96,
                        output_window_size: int = 96,
                        horizon: int = 1,
                        stride: int = 1,
                        scaler: Scale = None):
    """
        Load the US CDC ILI dataset and return it in SSTDataset format.

        :param us_cdc_ili_data_root: Root directory of the US CDC ILI dataset.
        :param vars: List of target variables. Defaults to None, selecting all variables.
        :param time_as_feature: Whether to include time as a feature. Defaults to False.
        :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
        :param input_window_size: Size of the input window. Defaults to 96.
        :param output_window_size: Size of the output window. Defaults to 96.
        :param horizon: Time steps between the input and output windows. Defaults to 1.
        :param stride: Time steps between consecutive windows. Defaults to 1.
        :param scaler: Scaler for the target variables. Defaults to None.
        :return: (training dataset, validation dataset), scaler
    """

    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(us_cdc_ili_data_root, 'US_National_ILI_1997_2025.csv')

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = [df.columns[-1]]

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)


"""

    WHO Japan ILI dataset.

"""


def get_who_japan_ili_columns(return_vars: Literal['univariate', 'multivariate', 'all'] = 'univariate'):
    """
        Get the columns of WHO Japan ILI dataset.

        :return: The columns of WHO Japan ILI dataset.
    """
    columns = ['Date', 'AH1', 'AH1N12009', 'AH3', 'AH5', 'ANOTSUBTYPED', 'INF_A', 'BVIC', 'BYAM', 'BNOTDETERMINED',
               'INF_B', 'INF_ALL', 'INF_NEGATIVE', 'ILI_ACTIVITY']

    if return_vars == 'univariate':
        return columns[-1:]
    elif return_vars == 'multivariate':
        return columns[1:]

    return columns


def load_who_japan_ili_sst(who_japan_ili_data_root: str,
                           vars: List[str] = None,
                           time_as_feature: TimeAsFeature = None,
                           split_ratio: float = 0.8,
                           input_window_size: int = 96,
                           output_window_size: int = 96,
                           horizon: int = 1,
                           stride: int = 1,
                           scaler: Scale = None):
    """
    Load the WHO Japan ILI dataset and return it in SSTDataset format.

    :param who_japan_ili_data_root: Root directory of the WHO Japan ILI dataset.
    :param vars: List of target variables. Defaults to None, selecting all variables.
    :param time_as_feature: Whether to include time as a feature. Defaults to False.
    :param split_ratio: Ratio for splitting the dataset into training and validation sets. Defaults to 0.8.
    :param input_window_size: Size of the input window. Defaults to 96.
    :param output_window_size: Size of the output window. Defaults to 96.
    :param horizon: Time steps between the input and output windows. Defaults to 1.
    :param stride: Time steps between consecutive windows. Defaults to 1.
    :param scaler: Scaler for the target variables. Defaults to None.
    :return: (training dataset, validation dataset), scaler
    """
    # Validate input parameters
    assert 0 < split_ratio <= 1, f"Invalid split ratio: {split_ratio}"
    assert input_window_size > 0, f"Invalid input window size: {input_window_size}"
    assert output_window_size > 0, f"Invalid output window size: {output_window_size}"
    assert stride > 0, f"Invalid stride: {stride}"

    # Path to the dataset CSV file
    csv_file = os.path.join(who_japan_ili_data_root, 'Japan_ILI_19961006_20250309.csv')

    # Check if the file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Load the dataset into a DataFrame
    print(f"Loading: {csv_file}")
    df = pd.read_csv(csv_file)

    # Initialize parameters for SSTDataset
    sst_params = {
        'ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    # Select target variables
    if vars is None:
        vars = [df.columns[-1]]

    # Extract target variables and convert to tensor
    target_df = df.loc[:, vars]
    target_array = target_df.values.astype('float32')
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    # Apply scaling if a scaler is provided
    if scaler is not None:
        scaler = scaler.fit(target_tensor)

    # Add time as a feature if specified
    if time_as_feature:
        df['Date'] = pd.to_datetime(df['Date'])
        time_features = df['Date'].dt
        time_feature_array = time_features.hour.values.reshape(-1, 1).astype('float32')  # Example: extract hour feature
        time_feature_tensor = torch.tensor(time_feature_array)
        sst_params['ex_ts2'] = time_feature_tensor

    # Handle dataset splitting
    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, None)

    train_ds = SSTDataset(**sst_params, stride=stride, split='train')
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, None)
