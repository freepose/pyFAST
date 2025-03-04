# !/usr/bin/env python
# encoding: utf-8

import os
import random
import numpy as np
import pandas as pd
import torch

from prepare_data_back import time_series_scaler
from fast.data import STMDataset, Scale, StandardScale, MinMaxScale, train_test_split, STSDataset


def load_battery_data(data_root: str, ds_names: list[str] = 'nenergy_2019', ds_params: dict = {},
                      split_ratio: float = 0.8,
                      use_features: bool = False, scaler: Scale = Scale(), target_factor: int = 100) -> tuple:
    """ Load and preprocess battery datasets for model training and evaluation."""
    dataset_paths = {
        'nenergy_2019': 'time_series/energy_battery/nenergy_2019_battery_cycle_life/02_ts_cycle/tiv2023_deephpm/',
        'hust': 'time_series/energy_battery/hust/2_ts_cycle',
        'dfdq': 'time_series/energy_battery/dfdq/2_uts/train',
        'calce': 'time_series/energy_battery/calce/3_cycle_data',
    }

    csv_list = []
    for ds_name in ds_names:
        csv_file_dir = os.path.join(data_root, dataset_paths[ds_name])
        csv_filenames = [filename for filename in os.listdir(csv_file_dir) if filename.endswith('.csv')]
        csv_filenames = [os.path.join(csv_file_dir, name) for name in csv_filenames]
        csv_list.extend(csv_filenames)

    target_ts_list, feature_ts_list = process_battery_data(csv_list, use_smooth=True)
    target_ts_list = [ts * target_factor for ts in target_ts_list]
    random.shuffle(target_ts_list)
    split_idx = int(len(target_ts_list) * split_ratio)
    train_target, val_target = target_ts_list[:split_idx], target_ts_list[split_idx:]
    train_feature, val_feature = feature_ts_list[:split_idx], feature_ts_list[split_idx:]

    train_target = [torch.tensor(ts, dtype=torch.float32) for ts in train_target]
    val_target = [torch.tensor(ts, dtype=torch.float32) for ts in val_target]

    train_feature = [torch.tensor(feat, dtype=torch.float32) for feat in train_feature]
    val_feature = [torch.tensor(feat, dtype=torch.float32) for feat in val_feature]

    train_ex_ts, val_ex_ts = (train_feature, val_feature) if use_features else (None, None)
    global_target_scaler = time_series_scaler(train_target, scaler)
    global_feature_scaler = time_series_scaler(train_ex_ts, scaler) if use_features else None

    train_ds = STMDataset(train_target, ex_ts=train_ex_ts, **ds_params, stride=1)
    val_ds = STMDataset(val_target, ex_ts=val_ex_ts, **ds_params, stride=ds_params['output_window_size'])

    return (train_ds, val_ds), (global_target_scaler, global_feature_scaler)


def process_battery_data(filenames: list[str],
                         target: str = 'SoH',
                         use_smooth: bool = False,
                         feature_columns: list[str] = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    通用电池数据集处理函数。

    :param filenames: 获取所有 CSV 文件路径。
    :param target: 目标列名，默认是 'SoH'。
    :param use_smooth: 是否对数据应用平滑处理。
    :param feature_columns: 手动指定的特征列集合（可选）。如果未指定，将从所有文件中计算特征列的并集。
    :return: 一个元组 (target_ts_list, feature_ts_list)，分别表示目标值时间序列和特征时间序列。
    """

    # 初始化
    df_list = []
    combined_columns = set()

    # 读取所有文件，计算特征列的并集
    for filename in sorted(filenames):
        df = pd.read_csv(filename)
        combined_columns.update(df.columns)
        df_list.append(df)

    # 如果未手动指定特征列，则自动确定
    if feature_columns is None:
        feature_columns = list(combined_columns - {target})

    target_ts_list, feature_ts_list = [], []

    # 遍历数据框列表，处理缺失列和特征时间序列
    for df in df_list:
        # 确保所有缺失列填充为 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        # 仅保留 target 值在 0.8 和 1 之间的行
        if target in df.columns:
            df = df[(df[target] >= 0.8) & (df[target] <= 1.0)].copy()
        # 应用平滑处理（如果启用）
        if use_smooth:
            df[feature_columns] = df[feature_columns].rolling(window=5, min_periods=1).mean().round(5)
            if target in df.columns:
                df[target] = df[target].rolling(window=5, min_periods=1).mean().round(5)

        # 提取目标值和特征值
        if target in df.columns:
            target_ts = df[target].values.reshape(-1, 1)
        else:
            target_ts = np.zeros((len(df), 1))

        feature_ts = df[feature_columns].values

        target_ts_list.append(target_ts)
        feature_ts_list.append(feature_ts)

    return target_ts_list, feature_ts_list

