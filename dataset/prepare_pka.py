#!/usr/bin/env python
# encoding: utf-8

"""
    Prepare Biological datasets.
    Biological datasets (pKa).
"""

import os, sys
from typing import Literal

import numpy as np
import pandas as pd

import torch

from tqdm import tqdm
from fast.data import SMTDataset, BDPDataset


def get_pKa_csv(csv_files: list, desc: str = None):
    """
        Read pKa data from csv files.
        :param csv_files: input csv files.
        :param desc: the description of the progress bar.
        :return: ts_list, ts_mask_list, feature_list.
    """
    ts_list, ts_mask_list, feature_list = [], [], []
    with tqdm(total=len(csv_files), leave=False, file=sys.stdout) as pbar:
        for csv_file in csv_files:
            pbar.set_description(desc)

            df = pd.read_csv(csv_file)
            ts_array = df['pKa shift'].values.reshape(-1, 1).astype(np.float32)
            ts_mask_array = (df['Res Name'].isna() == False).values.reshape(-1, 1)
            feature_array = df.loc[:, '0':'479'].values.astype(np.float32)

            ts_tensor = torch.tensor(ts_array)
            ts_mask_tensor = torch.tensor(ts_mask_array)
            feature_tensor = torch.tensor(feature_array)

            ts_list.append(ts_tensor)
            ts_mask_list.append(ts_mask_tensor)
            feature_list.append(feature_tensor)

            pbar.update(1)

    return ts_list, ts_mask_list, feature_list


def load_pka(data_root: str, ds_params: dict,
             ds_name: Literal['phmd_2d_549', 'deepka_5w'] = 'phmd_2d_549',
             load_as: Literal['bdp', 'smt'] = 'bdp') -> tuple:
    """
        Load pKa datasets to ``SMTDataset`` or ``BDPDataset``.

        The data fields are:
        [PDB ID,chain,amino acid,Res Name,Res ID,Titration,Target_pKa,model_pKa,pKa shift,res_name, 0 - 479 embeddings]

        :param data_root: the root directory of the whole datasets.
        :param ds_params: the dataset parameters.
        :param ds_name: ['phmd_2d_549', 'deepka_5w']
        :param load_as: the dataset type, either 'stm' or 'bdp'.
        :return: train and validation datasets.
    """
    pka_data_root = data_root + '/protein_pKa/{}/'.format(ds_name)
    dirs = {'train': pka_data_root + 'train', 'val': pka_data_root + 'test'}

    ds_cls_dict = {'smt': SMTDataset, 'bdp': BDPDataset}
    ds_cls = ds_cls_dict.get(load_as, BDPDataset)
    if load_as == 'smt':
        ds_params.update({'split_ratio': 1, 'split': 'train'})

    train_csv_files = [os.path.join(dirs['train'], f) for f in sorted(os.listdir(dirs['train'])) if f.endswith('.csv')]
    val_csv_files = [os.path.join(dirs['val'], f) for f in sorted(os.listdir(dirs['val'])) if f.endswith('.csv')]

    # train_csv_files = train_csv_files[:15]
    # val_csv_files = val_csv_files[:2]

    train_ts, train_ts_mask, train_features = get_pKa_csv(train_csv_files, 'Loading train')
    val_ts, val_ts_mask, val_features = get_pKa_csv(val_csv_files, 'Loading val')

    train_ds = ds_cls(train_ts, train_ts_mask, train_features, **ds_params)
    ds_params.update({'stride': ds_params['output_window_size']})
    val_ds = ds_cls(val_ts, val_ts_mask, val_features, **ds_params)

    return train_ds, val_ds
