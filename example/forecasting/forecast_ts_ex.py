#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on single time series forecasting using exogenous time series.

    (1) Ensemble Time series Learning for Time Series Forecasting.

"""

import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import scaler_fit, scaler_transform, StandardScale, MinMaxScale, AbstractScale
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_weight_types
from fast.model.mts_fusion import ARX, NARXMLP, NARXRNN
from fast.model.mts_fusion import DSAR, DGR, DGDR, MvT, GAINGE, TSPT

from dataset.prepare_xmcdc import load_xmcdc_as_sst, load_xmcdc_as_smt
from dataset.manage_sst_datasets import prepare_sst_datasets
from dataset.manage_smx_datasets import prepare_smx_datasets


def main():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'mps'

    xmcdc_filename = '../../dataset/xmcdc/outpatients_2011_2020_1day.csv'
    # train_ds, val_ds, test_ds = load_xmcdc_as_sst(xmcdc_filename, None, False, ['bsi'], False, 10, 1, 1, 1, (0.7, 0.1, 0.2), ds_device)
    # train_ds, val_ds, test_ds = load_xmcdc_as_smt(xmcdc_filename, None, False, ['bsi'], False, 10, 1, 1, 1, (0.7, 0.1, 0.2), ds_device)

    task_config = {'ts': 'multivariate', 'use_ex': True}
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'SuzhouIPL', 8 * 24, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_smx_datasets(data_root, 'GreeceWPF', 10 * 24, 1 * 24, 1, 1, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smx_datasets(data_root, 'SDWPF', 6 * 24, 6 * 6, 1, 1, (0.7, 0.1, 0.2), 'inter', ds_device, **task_config)
    train_ds, val_ds, test_ds = prepare_smx_datasets(data_root, 'GFM', 5, 3, 1, 3, (0.7, 0.1, 0.2), 'inter', ds_device, **task_config)

    overwrite_scaler = scaler_fit(StandardScale(), train_ds.ts)
    # train_ds.ts = scaler_transform(overwrite_scaler, train_ds.ts)
    overwrite_ex_scaler = scaler_fit(StandardScale(), train_ds.ex_ts) if train_ds.ex_ts is not None else None
    # if val_ds is not None:
        # val_ds.ts = scaler_transform(overwrite_scaler, val_ds.ts)
        # val_ds.ex_ts = scaler_transform(overwrite_ex_scaler, val_ds.ex_ts) if val_ds.ex_ts is not None else None
    # if test_ds is not None:
        # test_ds.ts = scaler_transform(overwrite_scaler, test_ds.ts)
        # test_ds.ex_ts = scaler_transform(overwrite_ex_scaler, test_ds.ex_ts) if test_ds.ex_ts is not None else None

    scaler, ex_scaler = None, overwrite_ex_scaler # overwrite_ex_scaler

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    modeler = {
        'arx': [ARX, {'ex_retain_window_size': train_ds.input_window_size}],
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

    model_cls, user_args = modeler['narx-rnn']

    common_ds_args = get_common_kwargs(model_cls.__init__, train_ds.__dict__)
    combined_args = {**common_ds_args, **user_args}
    model = model_cls(**combined_args)

    model = covert_weight_types(model, torch_float_type)
    print(get_model_info(model))

    model_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_weights, lr=0.0005, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel', verbose=False)

    criterion = MSE()
    evaluator = Evaluator(['MSE', 'RMSE', 'MAE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)
    logging.getLogger().info(f"{trainer}")

    batch_size = 512
    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 200), batch_size=batch_size, shuffle=True,
                verbose=2)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, batch_size, None, False, is_online=False)
        print('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, batch_size, None, False, is_online=False)
        print('val {}'.format(results))

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    initial_logger()

    main()
