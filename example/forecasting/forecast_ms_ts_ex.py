#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on **multiple sources** time series forecasting using exogenous variables.

    (1) NARX: Using exogenous variables as input features.

    (2) Sparse NARX: Using exogenous **sparse** variables as input features.

"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_common_params
from fast.train import Trainer
from fast.metric import Evaluator

from fast.model.base import count_parameters, covert_parameters
from fast.model.mts_fusion import ARX, NARXMLP, NARXRNN
from fast.model.mts_fusion import DSAR, DGR, DGDR, MvT, GAINGE
from fast.model.mts_fusion import SparseNARXRNN

from dataset.prepare_xmcdc import load_xmcdc_smt
from dataset.prepare_industrial_power_load import load_industrial_power_load_smt as load_ipl_smt


def ms_ts_fusion():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    (train_ds, val_ds), (scaler, ex_scaler) = load_xmcdc_smt('1week', None, ['weather'], **ds_params)

    # ds_params = {'input_window_size': 8 * 24, 'output_window_size': 24, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_ipl_smt(data_root, ex_vars=['temperature', 'humidity'], **ds_params)

    # ds_params = {'input_window_size': 6 * 4, 'output_window_size': 4, 'horizon': 1}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_diabetes_smt(data_root, 'all', ds_params, True, 'inter', 0.8)

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
        'sparse-narx-rnn': [SparseNARXRNN, {'rnn_cls': 'rnn', 'hidden_size': 64, 'num_layers': 1,
                                            'bidirectional': False, 'dropout_rate': 0.}],
    }

    model_cls, user_settings = modeler['dgdr']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print('{}\n{}\n{}'.format(train_ds, val_ds, model))

    model_name = type(model).__name__
    model = covert_parameters(model, torch_float_type)
    print(model_name, count_parameters(model))

    criterion = nn.MSELoss()
    additive_criterion = getattr(model, 'loss', None)
    evaluator = Evaluator(['MAE', 'RMSE', 'PCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      criterion=criterion, additive_criterion=additive_criterion, evaluator=evaluator,
                      global_scaler=scaler, global_ex_scaler=ex_scaler)

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=4, shuffle=False,
                verbose=True, display_interval=0)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    ms_ts_fusion()
