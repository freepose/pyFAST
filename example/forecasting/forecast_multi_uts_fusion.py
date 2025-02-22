#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on multivariate time series (MTS) forecasting using exogenous variables.

    (1) PINN: Using time variables as exogenous variables.

    (2) NARX: Using exogenous variables as input features.
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_common_params
from fast.data import Scale, MinMaxScale, StandardScale
from fast.train import Trainer
from fast.metric import Evaluator, mean_absolute_scaled_error

from fast.model.base import count_parameters, covert_parameters
from fast.model.mts_fusion import ARX, NARXMLP, NARXRNN
from fast.model.mts_fusion import DSAR, DGR, DGDR, MvT, GAINGE

from example.prepare_data import load_xmcdc_cases
from example.prepare_data import load_grid_forming_converter
from example.prepare_data import load_greek_wind, load_kddcup2022_sdwpf


def mts_fusion():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1}
    (train_ds, val_ds), (scaler, ex_scaler) = load_xmcdc_cases('../../dataset/xmcdc/', 'daily', ds_params, 0.8, True)

    # ds_params = {'input_window_size': 10 * 24, 'output_window_size': 12, 'horizon': 1}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_greek_wind(data_root, '1hour', ds_params, 1., 0.8, True)

    # ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_greek_wind(data_root, '1day', ds_params, 0.001, 0.8, True)

    # ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_kddcup2022_sdwpf(data_root, '1day', ds_params, 0.001, 0.8, True)

    # ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_grid_forming_converter(data_root, ds_params, 0.8, True)

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
    }

    model_cls, user_settings = modeler['dgr']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print('{}\n{}\n{}'.format(train_ds, val_ds, model))

    model_name = type(model).__name__
    model = covert_parameters(model, torch_float_type)
    print(model_name, count_parameters(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)

    criterion = nn.MSELoss()
    additive_criterion = getattr(model, 'loss', None)
    evaluator = Evaluator(['MAE', 'RMSE', 'CV-RMSE'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, additive_criterion=additive_criterion, evaluator=evaluator,
                      global_scaler=scaler, global_ex_scaler=ex_scaler)

    trainer.fit(train_ds,
                val_ds,
                epoch_range=(1, 2000), batch_size=512, shuffle=True,
                verbose=True, display_interval=0)

    print('Good luck!')


if __name__ == '__main__':
    mts_fusion()
