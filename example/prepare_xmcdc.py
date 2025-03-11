#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare datasets by loading XMCDC case counts as ``STSDataset`` and ``STMDataset``.
"""

from typing import Literal, List

import numpy as np
import pandas as pd

import torch

from fast.data import Scale, StandardScale, MinMaxScale, train_test_split, time_series_scaler
from fast.data import STSDataset, STMDataset


def load_xmcdc_sts(data_root: str = '../dataset/xmcdc/',
                   freq: Literal['daily', 'weekly'] = 'daily',
                   targets: List[str] = None,
                   exogeneities: List[str] = None,
                   ds_params: dict = None,
                   scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load XMCDC disease inpatient case count as ``STSDataset`` for mts forecasting (using exogenous data).

        The data fields are as follows:
        For daily frequency:
        ['Date', '平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点', '最高湿度(%)',
       '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压', '最高风速', '平均风速', '最低风速',
       'BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise', 'BSI_厦门肝炎_all',
       'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise', 'BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc',
       'BSI_厦门腹泻_wise', '手足口病', '肝炎', '其他感染性腹泻']

        For weekly frequency:
        ['Date', '平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点', '最高湿度(%)',
       '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压', '最高风速', '平均风速', '最低风速',
       'BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise', 'BSI_厦门肝炎_all',
       'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise', 'BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc',
       'BSI_厦门腹泻_wise', '手足口病', '肝炎', '其他感染性腹泻', 'HF1', 'HF2', 'HF3', 'HF4',
       'HF5', 'HF6', 'HF7', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'D1',
       'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

        :param data_root: the root directory of the whole datasets.
        :param freq: the frequency of the time series, either 'daily' or 'weekly'.
        :param targets: the list of disease names, one or all in ['手足口病', '肝炎', '其他感染性腹泻']. The default is all.
        :param exogeneities: the list of exogenous factors,
                            maybe none, one or all in ['weather', 'bsi', 'fine_grained', 'all'].
                            The 'fine_grained' is available only for weekly frequency.
        :param ds_params: the common dataset parameters for train and test ``STSDataset``.
                          Use ``split_ratio`` to split the target time series into train and test.
        :param scaler: the global scaler target time series and exogenous time series, default is Scale().
        :return: train and validation dataset, and the target and exogenous scalers.
    """
    if targets is None:
        targets = ['手足口病', '肝炎', '其他感染性腹泻']

    csv_file = data_root + r'{}_outpatients_2011_2020.csv'.format(freq)
    df = pd.read_csv(csv_file)

    # time_df = df.loc[:, 'Date':'Date']

    target_df = df.loc[:, targets]
    target_array = target_df.values.astype(np.float32)
    target_tensor = torch.tensor(target_array)
    target_scaler = time_series_scaler(target_tensor, scaler)

    ex_tensor = None
    ex_scaler = None
    if exogeneities is not None:
        weather_columns = ['平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                           '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压',
                           '最高风速', '平均风速', '最低风速']
        target_bsi_lookup = {'手足口病': ['BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise'],
                             '肝炎': ['BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise'],
                             '其他感染性腹泻': ['BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise']}
        target_fine_grained_lookup = {'手足口病': ['HF{}'.format(i) for i in range(1, 8)],
                                      '肝炎': ['H{}'.format(i) for i in range(1, 8)],
                                      '其他感染性腹泻': ['D{}'.format(i) for i in range(1, 8)]}

        ex_columns = weather_columns if 'weather' in exogeneities else []
        for t in targets:
            for ex in exogeneities:
                if ex == 'bsi':
                    ex_columns += target_bsi_lookup[t]
                elif ex == 'fine_grained' and freq == 'weekly':
                    ex_columns += target_fine_grained_lookup[t]
                elif ex == 'all':
                    ex_columns += target_bsi_lookup[t] + target_fine_grained_lookup[t]

        ex_df = df.loc[:, ex_columns]
        ex_array = ex_df.values.astype(np.float32)
        ex_tensor = torch.tensor(ex_array)

        ex_scaler = time_series_scaler(ex_array, scaler)

    params = {'ts': target_tensor, 'ts_mask': None, 'ex_ts': ex_tensor}
    params.update(ds_params)

    train_ds = STSDataset(**params, stride=1, split='train')
    val_ds = STSDataset(**params, stride=ds_params['output_window_size'], split='val')

    return (train_ds, val_ds), (target_scaler, ex_scaler)


def load_xmcdc_stm(data_root: str = '../dataset/xmcdc/',
                   freq: Literal['daily', 'weekly'] = 'daily',
                   targets: List[str] = None,
                   exogeneities: List[str] = None,
                   ds_params: dict = None,
                   scaler: Scale = Scale()) -> tuple[tuple, tuple]:
    """
        Load XMCDC disease inpatient case count as ``STMDataset`` for multi-source time series forecasting (using
        exogenous data).

        All diseases are used as target time series. There are three diseases: '手足口病', '肝炎', '其他感染性腹泻'.

        :param data_root: the root directory of the whole datasets.
        :param freq: the frequency of the time series, either 'daily' or 'weekly'.
        :param targets: the list of disease names, one or all in ['手足口病', '肝炎', '其他感染性腹泻']. The default is all.
        :param exogeneities: the list of exogenous factors,
                            maybe none, one or all in ['weather', 'bsi', 'fine_grained', 'all'].
                            The 'fine_grained' is available only for weekly frequency.
        :param ds_params: the common dataset parameters for train and test ``STSDataset``.
                          Use ``split_ratio`` to split the target time series into train and test.
        :param scaler: the global scaler target time series and exogenous time series, default is Scale().
        :return: train and validation dataset, and the target and exogenous scalers.
    """
    if targets is None:
        targets = ['手足口病', '肝炎', '其他感染性腹泻']

    csv_file = data_root + r'{}_outpatients_2011_2020.csv'.format(freq)
    df = pd.read_csv(csv_file)

    weather_columns = ['平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                       '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压',
                       '最高风速', '平均风速', '最低风速']
    target_bsi_lookup = {'手足口病': ['BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise'],
                         '肝炎': ['BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise'],
                         '其他感染性腹泻': ['BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise']}
    target_fine_grained_lookup = {'手足口病': ['HF{}'.format(i) for i in range(1, 8)],
                                  '肝炎': ['H{}'.format(i) for i in range(1, 8)],
                                  '其他感染性腹泻': ['D{}'.format(i) for i in range(1, 8)]}

    ts_list = []
    ex_ts_list = [] if exogeneities is not None else None
    for name in targets:
        ts_array = df[name].values.reshape(-1, 1).astype(np.float32)
        ts_tensor = torch.tensor(ts_array)
        ts_list.append(ts_tensor)

        if exogeneities is not None:
            ex_columns = weather_columns if 'weather' in exogeneities else []
            for ex in exogeneities:
                if ex == 'bsi':
                    ex_columns += target_bsi_lookup[name]
                elif ex == 'fine_grained' and freq == 'weekly':
                    ex_columns += target_fine_grained_lookup[name]
                elif ex == 'all':
                    ex_columns += target_bsi_lookup[name] + target_fine_grained_lookup[name]

            ex_df = df.loc[:, ex_columns]
            ex_array = ex_df.values.astype(np.float32)
            ex_tensor = torch.tensor(ex_array)

            ex_ts_list.append(ex_tensor)

    target_scaler = time_series_scaler(ts_list, scaler)
    ex_scaler = time_series_scaler(ex_ts_list, scaler) if exogeneities is not None else None

    params = {'ts': ts_list, 'ex_ts': ex_ts_list}
    params.update(ds_params)

    train_ds = STMDataset(**params, stride=1, split='train')
    val_ds = STMDataset(**params, stride=ds_params['output_window_size'], split='val')

    return (train_ds, val_ds), (target_scaler, ex_scaler)
