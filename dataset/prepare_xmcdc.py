#!/usr/bin/env python
# encoding: utf-8

"""
    This is an example of how to prepare the XMCDC datasets for forecasting tasks using built-in datasets.

    Prepare datasets by loading XMCDC case counts as ``SSTDataset`` and ``SMTDataset``.
"""

import os, logging

import numpy as np
import pandas as pd
import torch

from typing import Literal, List, Union, Tuple
from fast.data import SSTDataset, SMTDataset

from fast.data.processing import load_sst_datasets, load_smx_datasets

SSTDatasetSequence = Union[SSTDataset, List[SSTDataset]]
SMTDatasetSequence = Union[SMTDataset, List[SMTDataset]]

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


def get_xmcdc_targets() -> List[str]:
    """
        Get the target variables of XMCDC datasets.

        :return: the list of target variables.
    """
    return ['手足口病', '肝炎', '其他感染性腹泻']


def get_xmcdc_ex_columns(freq: Literal['1day', '1week'] = '1day', targets: List[str] = None,
                         ex_category: Literal['weather', 'bsi', 'fine_granularity', 'all'] = 'weather') -> List[str]:
    """
        The XMCDC datasets have too many columns, and this function is used to get the columns by category.

        :param freq: the frequency of the time series, either '1day' or '1week'.
        :param targets: the target variables, can be one or more in ['手足口病', '肝炎', '其他感染性腹泻'].
        :param ex_category: the category of the exogenous variables,
                            can be 'weather', 'bsi', 'fine_granularity', or 'all'.
                            The default is 'weather', and can not be ``None``.
                            If 'all', then all exogenous variables are included.

        :return: the selected column names.
    """

    assert freq in ['1day', '1week'], "Frequency must be either '1day' or '1week'."
    assert ex_category in ['weather', 'bsi', 'fine_granularity', 'all'], \
        "Exogenous category must be one of 'weather', 'bsi', 'fine_granularity', or 'all'."

    if targets is None:
        targets = ['手足口病', '肝炎', '其他感染性腹泻']

    weather_columns = ['平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                       '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压',
                       '最高风速', '平均风速', '最低风速']

    target_bsi_lookup = {
        '手足口病': ['BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise'],
        '肝炎': ['BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise'],
        '其他感染性腹泻': ['BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise']
    }

    target_fine_grained_lookup = {
        '手足口病': ['HF{}'.format(i) for i in range(1, 8)],
        '肝炎': ['H{}'.format(i) for i in range(1, 8)],
        '其他感染性腹泻': ['D{}'.format(i) for i in range(1, 8)]
    }

    ret_columns = []
    if ex_category in ('weather', 'all'):
        ret_columns.extend(weather_columns)

    if ex_category in ('bsi', 'all'):
        for target in targets:
            ret_columns.extend(target_bsi_lookup[target])

    if freq == '1week' and ex_category in ('fine_granularity', 'all'):
        for target in targets:
            ret_columns.extend(target_fine_grained_lookup[target])

    return ret_columns


def load_xmcdc_as_sst(filename: str = '../../dataset/xmcdc/outpatients_2011_2020_1day.csv',
                      variables: List[str] = None,
                      mask_variables: bool = False,
                      ex_categories: List[str] = None,
                      mask_ex_variables: bool = False,
                      input_window_size: int = 10,
                      output_window_size: int = 1,
                      horizon: int = 1,
                      stride: int = 1,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu') -> SSTDatasetSequence:
    """
        Load XMCDC disease outpatient count as ``SSTDataset``.

        Load time series dataset from a **CSV** file, transform time series data into supervised data,
        and split the dataset into training, validation, and test sets.

        The default **float type** is ``float32``, you can change it to ``float64`` if needed.
        The default **device** is ``cpu``, you can change it to ``cuda`` or ``mps`` if needed.

        :param filename: the CSV filename.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series.
        :param ex_categories: categories of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.

        :return: the (split) datasets as SSTDataset objects.
    """

    if variables is None:
        variables = get_xmcdc_targets()

    ex_variables = None
    freq = filename.split('_')[-1].strip('.csv')
    if ex_categories is not None:
        ex_variables = []
        for category in ex_categories:
            for target in variables:
                ex_variables.extend(get_xmcdc_ex_columns(freq, [target], category))

    logging.getLogger().info('Loading {}'.format(filename))
    sst_datasets = load_sst_datasets(filename, variables, mask_variables, ex_variables, mask_ex_variables, None,
                                     input_window_size, output_window_size, horizon, stride, split_ratios, device)

    return sst_datasets


def load_xmcdc_as_smt(filename: str = '../../dataset/xmcdc/outpatients_2011_2020_1day.csv',
                      variables: List[str] = None,
                      mask_variables: bool = False,
                      ex_categories: List[str] = None,
                      mask_ex_variables: bool = False,
                      input_window_size: int = 96,
                      output_window_size: int = 24,
                      horizon: int = 1,
                      stride: int = 1,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu') -> SMTDatasetSequence:
    """
        Load XMCDC time series as **SMTDataset**s.

        :param filename: the CSV filename.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_categories: categories of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param input_window_size: input window size of the transformed supervised data
                            A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data
                            A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70 % for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.

        :return: the (split) datasets as SMTDataset objects.
    """

    if variables is None:
        variables = get_xmcdc_targets()

    freq = filename.split('_')[-1].strip('.csv')

    sst_datasets = []
    for target in variables:
        ex_variables = None
        if ex_categories is not None:
            ex_variables = []
            for category in ex_categories:
                ex_variables.extend(get_xmcdc_ex_columns(freq, [target], category))

        logging.getLogger().info('Loading {}'.format(filename))
        sst = load_sst_datasets(filename, [target], mask_variables, ex_variables, mask_ex_variables, None,
                                input_window_size, output_window_size, horizon, stride, None, device)
        sst_datasets.append(sst)

    # Merge all SSTDatasets into a single SMTDataset
    smt_args = dict({
        'ts': [],
        'ts_mask': [] if mask_variables else None,
        'ex_ts': [] if ex_categories is not None else None,
        'ex_ts_mask': [] if (ex_categories is not None and mask_ex_variables) else None,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
        'stride': stride,
    })

    for sst in sst_datasets:
        smt_args['ts'].append(sst.ts)
        if mask_variables:
            smt_args['ts_mask'].append(sst.ts_mask)
        if ex_categories is not None:
            smt_args['ex_ts'].append(sst.ex_ts)
            if mask_ex_variables:
                smt_args['ex_ts_mask'].append(sst.ex_ts_mask)

    if split_ratios is None:
        return SMTDataset(**smt_args)

    smt_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])
    for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
        if i > 0:
            smt_args.update({'stride': output_window_size})
        split_ds = SMTDataset(**smt_args).split(s, e, is_strict=False, mark='split_{}'.format(i))
        smt_datasets.append(split_ds)

    return smt_datasets
