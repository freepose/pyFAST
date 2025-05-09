#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on single source univariate / multivariate time series forecasting using exogenous time series.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_device, get_common_params
from fast.data import MinMaxScale
from fast.train import Trainer
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_parameters
from fast.model.mts_fusion import ARX, NARXMLP, NARXRNN
from fast.model.mts_fusion import DSAR, DGR, DGDR, MvT, GAINGE, TSPT

from dataset.prepare_xmcdc import load_xmcdc_sst
from dataset.prepare_industrial_power_load import load_industrial_power_load_sst as load_ipl_sst


def mts_fusion():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    device = get_device('cpu')

    # ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_xmcdc_sst('1week', None, ['weather'], **ds_params)

    ds_params = {'input_window_size': 8 * 24, 'output_window_size': 24, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    (train_ds, val_ds), (scaler, ex_scaler) = load_ipl_sst(data_root, ex_vars=['temperature', 'humidity'], **ds_params)

    modeler = {
        'arx': [ARX, {'ex_retain_window_size': train_ds.input_window_size // 2}],
        'narx-mlp': [NARXMLP, {'ex_retain_window_size': train_ds.input_window_size // 2,
                               'hidden_units': [32], 'activation': 'linear'}],
        'narx-rnn': [NARXRNN, {'rnn_cls': 'rnn', 'hidden_size': 64, 'num_layers': 1,
                               'bidirectional': False, 'dropout_rate': 0.}],
        'dsar': [DSAR, {'ex_retain_window_size': train_ds.input_window_size // 2, 'dropout_rate': 0.}],
        'dgr': [DGR, {'ex_retain_window_size': train_ds.input_window_size,
                      'rnn_cls': 'rnn', 'hidden_size': 256, 'ex_hidden_size': 32, 'num_layers': 3,
                      'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
        'dgdr': [DGDR, {'dropout_rate': 0.}],
        'mvt': [MvT, {'ex_retain_window_size': train_ds.input_window_size // 2, 'dropout_rate': 0.}],
        'gainge': [GAINGE, {'gat_h_dim': 4, 'dropout_rate': 0.01, 'highway_window_size': 7}],
        'tspt': [TSPT, {'ex_linear_layers': [32], 'target_linear_layers': [32],
                        'variable_hidden_size': 16, 'patch_len': 24, 'patch_stride': 24, 'patch_padding': 0,
                        'num_layers': 3, 'num_heads': 4, 'd_model': 64, 'dim_ff': 128,
                        'd_k': None, 'd_v': None, 'dropout_rate': 0.1,
                        'use_instance_scale': True}],
    }

    model_cls, user_settings = modeler['tspt']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print('{}\n{}'.format(train_ds, val_ds))

    model = covert_parameters(model, torch_float_type)
    print(get_model_info(model))

    model_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_weights, lr=0.0005, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)

    criterion = MSE()
    evaluator = Evaluator(['MAE', 'RMSE', 'PCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=512, shuffle=True,
                verbose=2)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    mts_fusion()
