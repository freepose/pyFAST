#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading XMCDC case counts as ``STSDataset`` and ``STMDataset``.
"""

from typing import Literal, List

import numpy as np
import pandas as pd

import torch

from fast.data import Scale, time_series_scaler
from fast.data import SSTDataset, SMTDataset

from dataset.time_feature import TimeAsFeature

"""

The data fields of XMCDC datasets are as follows:

(1) For daily frequency:
      ['Date', '平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点', '最高湿度(%)',
       '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压', '最高风速', '平均风速', '最低风速',
       'BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise', 
       'BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise',
       'BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc','BSI_厦门腹泻_wise', 
       '手足口病', '肝炎', '其他感染性腹泻']

(2) For weekly frequency:
      ['Date', '平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点', '最高湿度(%)',
       '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压', '最高风速', '平均风速', '最低风速',
       'BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise', 
       'BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise',
       'BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc','BSI_厦门腹泻_wise', 
       '手足口病', '肝炎', '其他感染性腹泻', 
       'HF1', 'HF2', 'HF3', 'HF4', 'HF5', 'HF6', 'HF7', 
       'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 
       'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
           
"""


def load_xmcdc_sst(freq: Literal['1day', '1week'] = '1day',
                   vars: List[str] = None,
                   ex_vars: List[str] = None,
                   use_time_features: bool = False,
                   split_ratio: float = 0.8,
                   input_window_size: int = 10,
                   output_window_size: int = 1,
                   horizon: int = 1,
                   stride: int = 1,
                   scaler: Scale = None,
                   ex_scaler: Scale = None) -> tuple[tuple, tuple]:
    """
        Load XMCDC disease outpatient count as ``SSTDataset`` for mts forecasting (or using exogenous data).

        :param freq: the frequency of the time series, either '1day' or '1week'.
        :param vars: target variables.
                     the list of disease names, one or all in ['手足口病', '肝炎', '其他感染性腹泻'].
                     If ``vars`` is None, then the value is all disease names.
        :param ex_vars: exogenous variables.
                        The list of exogenous factor types,
                        maybe none, one or all in ['weather', 'bsi', 'fine_grained', 'all'].
                        The 'fine_grained' is available only for weekly frequency.
                     If ``ex_vars`` is None, then exogenous factors are not used.
        :param use_time_features: whether to use time features, default is False.
        :param split_ratio: the ratio to split the target time series into train and test.
        :param input_window_size: the size of the input window, default is 10.
        :param output_window_size: the size of the output window, default is 1.
        :param horizon: the size of the horizon, default is 1.
        :param stride: the stride of the dataset, default is 1.
        :param scaler: the global scaler target time series and exogenous time series.
        :param ex_scaler: the global scaler for exogenous time series.
        :return: train and validation dataset, and the target and exogenous scalers.
    """

    sst_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'split_ratio': split_ratio,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
        'stride': stride,
    }

    if vars is None:
        vars = ['手足口病', '肝炎', '其他感染性腹泻']

    csv_file = r'../../dataset/xmcdc/outpatients_2011_2020_{}.csv'.format(freq)
    df = pd.read_csv(csv_file)

    target_df = df.loc[:, vars]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    sst_params['ts'] = target_tensor

    if scaler is not None and type(scaler) != type(Scale()):
        scaler = scaler.fit(target_tensor)

    if use_time_features:
        if freq == '1day':
            df['Date'] = pd.to_datetime(df['Date'])
            time_dt = df['Date'].dt
            time_feature_array = TimeAsFeature(freq='d', is_normalized=True)(time_dt)

            time_feature_array = time_feature_array.astype(np.float32)
            time_feature_tensor = torch.tensor(time_feature_array)
            sst_params['ex_ts2'] = time_feature_tensor

        elif freq == '1week':
            # 2020Y12W -> 52: extract week number
            weekofyear_series = df['Date'].apply(lambda x: int(x.split('Y')[1].rstrip('W')))
            weekofyear = weekofyear_series.values.astype(np.float32).reshape(-1, 1)
            weekofyear = (weekofyear - 1) / 52 - 0.5
            time_feature_tensor = torch.tensor(weekofyear)
            sst_params['ex_ts2'] = time_feature_tensor

    if ex_vars is not None:
        weather_columns = ['平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                           '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压',
                           '最高风速', '平均风速', '最低风速']
        target_bsi_lookup = {'手足口病': ['BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise'],
                             '肝炎': ['BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise'],
                             '其他感染性腹泻': ['BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise']}
        target_fine_grained_lookup = {'手足口病': ['HF{}'.format(i) for i in range(1, 8)],
                                      '肝炎': ['H{}'.format(i) for i in range(1, 8)],
                                      '其他感染性腹泻': ['D{}'.format(i) for i in range(1, 8)]}

        ex_columns = weather_columns if 'weather' in ex_vars else []
        for t in vars:
            for ex in ex_vars:
                if ex == 'bsi':
                    ex_columns += target_bsi_lookup[t]
                elif ex == 'fine_grained' and freq == 'weekly':
                    ex_columns += target_fine_grained_lookup[t]
                elif ex == 'all':
                    ex_columns += target_bsi_lookup[t] + target_fine_grained_lookup[t]

        ex_df = df.loc[:, ex_columns]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)
        sst_params['ex_ts'] = ex_tensor

        if ex_scaler is not None and type(ex_scaler) != type(Scale()):
            ex_scaler = ex_scaler.fit(ex_tensor)

    if split_ratio == 1.0:
        train_ds = SSTDataset(**sst_params, split='train')
        return (train_ds, None), (scaler, ex_scaler)

    train_ds = SSTDataset(**sst_params, split='train')
    del sst_params['stride']
    val_ds = SSTDataset(**sst_params, stride=sst_params['output_window_size'], split='val')

    return (train_ds, val_ds), (scaler, ex_scaler)


def load_xmcdc_smt(freq: Literal['1day', '1week'] = '1day',
                   vars: List[str] = None,
                   ex_vars: List[str] = None,
                   use_time_features: bool = False,
                   split_ratio: float = 0.8,
                   input_window_size: int = 10,
                   output_window_size: int = 1,
                   horizon: int = 1,
                   stride: int = 1,
                   scaler: Scale = None,
                   ex_scaler: Scale = None) -> tuple[tuple, tuple]:
    """
        Load XMCDC disease outpatient count as ``SMTDataset`` for multi-source time series forecasting (using
        exogenous data).

        :param freq: the frequency of the time series, either '1day' or '1week'.
        :param vars: target variables.
                     the list of disease names, one or all in ['手足口病', '肝炎', '其他感染性腹泻'].
                     If ``vars`` is None, then the value is all disease names.
        :param ex_vars: exogenous variables.
                        The list of exogenous factor types,
                        maybe none, one or all in ['weather', 'bsi', 'fine_grained', 'all'].
                        The 'fine_grained' is available only for weekly frequency.
                     If ``ex_vars`` is None, then exogenous factors are not used.
        :param use_time_features: whether to use time features, default is False.
        :param split_ratio: the ratio to split the target time series into train and test.
        :param input_window_size: the size of the input window, default is 10.
        :param output_window_size: the size of the output window, default is 1.
        :param horizon: the size of the horizon, default is 1.
        :param stride: the stride of the dataset, default is 1.
        :param scaler: the global scaler target time series and exogenous time series.
        :param ex_scaler: the global scaler for exogenous time series.
        :return: train and validation dataset, and the target and exogenous scalers.
    """

    weather_columns = ['平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                       '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压',
                       '最高风速', '平均风速', '最低风速']
    target_bsi_lookup = {'手足口病': ['BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise'],
                         '肝炎': ['BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise'],
                         '其他感染性腹泻': ['BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise']}
    target_fine_grained_lookup = {'手足口病': ['HF{}'.format(i) for i in range(1, 8)],
                                  '肝炎': ['H{}'.format(i) for i in range(1, 8)],
                                  '其他感染性腹泻': ['D{}'.format(i) for i in range(1, 8)]}

    if vars is None:
        vars = ['手足口病', '肝炎', '其他感染性腹泻']

    stm_params = {
        'ts': None,
        'ex_ts': None,
        'ex_ts2': None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
    }

    csv_file = r'../../dataset/xmcdc/outpatients_2011_2020_{}.csv'.format(freq)
    df = pd.read_csv(csv_file)

    ts_list = []
    ex_ts_list = [] if ex_vars is not None else None
    ex_ts2_list = [] if use_time_features else None

    for name in vars:
        ts_array = df[name].values.reshape(-1, 1).astype(np.float32)
        ts_tensor = torch.tensor(ts_array)
        ts_list.append(ts_tensor)

        if ex_vars is not None:
            ex_columns = weather_columns if 'weather' in ex_vars else []
            for ex in ex_vars:
                if ex == 'bsi':
                    ex_columns += target_bsi_lookup[name]
                elif ex == 'fine_grained' and freq == '1week':
                    ex_columns += target_fine_grained_lookup[name]
                elif ex == 'all':
                    ex_columns += target_bsi_lookup[name] + target_fine_grained_lookup[name]

            ex_df = df.loc[:, ex_columns]
            ex_array = ex_df.values.astype(np.float32)
            ex_tensor = torch.tensor(ex_array)
            ex_ts_list.append(ex_tensor)

        if use_time_features:
            if freq == '1day':
                df['Date'] = pd.to_datetime(df['Date'])
                time_dt = df['Date'].dt
                time_feature_array = TimeAsFeature(freq='d', is_normalized=True)(time_dt)

                time_feature_array = time_feature_array.astype(np.float32)
                time_feature_tensor = torch.tensor(time_feature_array)
                ex_ts2_list.append(time_feature_tensor)

            elif freq == '1week':
                weekofyear_series = df['Date'].apply(lambda x: int(x.split('Y')[1].rstrip('W')))
                weekofyear = weekofyear_series.values.astype(np.float32).reshape(-1, 1)
                weekofyear = (weekofyear - 1) / 52 - 0.5
                time_feature_tensor = torch.tensor(weekofyear)
                ex_ts2_list.append(time_feature_tensor)

    if scaler is not None and type(scaler) != type(Scale()):
        scaler = time_series_scaler(ts_list, scaler)

    if ex_vars is not None and ex_scaler is not None and type(ex_scaler) != type(Scale()):
        ex_scaler = time_series_scaler(ex_ts_list, ex_scaler)

    stm_params['ts'] = ts_list
    stm_params['ex_ts'] = ex_ts_list
    stm_params['ex_ts2'] = ex_ts2_list

    if split_ratio == 1.:
        train_ds = SMTDataset(**stm_params, stride=stride)
        return (train_ds, None), (scaler, ex_scaler)

    train_ds = SMTDataset(**stm_params, stride=stride).split(split_ratio, 'train', 'train')
    val_ds = SMTDataset(**stm_params, stride=stm_params['output_window_size']).split(split_ratio, 'val', 'val')

    return (train_ds, val_ds), (scaler, ex_scaler)
