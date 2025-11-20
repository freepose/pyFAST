#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on single time series forecasting using preknown exogenous time series, such as weather, time features.
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
from fast.model.mts_fusion import TransformerEx2, InformerEx2, AutoformerEx2, FEDformerEx2, DeepTIMe

from dataset.manage_sst_datasets import prepare_sst_datasets
from dataset.manage_smt_datasets import prepare_smt_datasets


def main():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'cpu'

    task_config = {'ts': 'multivariate', 'use_ex2': True}

    train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'ETTm2', 512, 96, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'ETTh1', 48, 24, 1, 1, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)

    overwrite_scaler = scaler_fit(StandardScale(), train_ds.ts)
    scaler_transform(overwrite_scaler, train_ds.ts, inplace=True)
    if val_ds is not None:
        scaler_transform(overwrite_scaler, val_ds.ts, inplace=True)
    if test_ds is not None:
        scaler_transform(overwrite_scaler, test_ds.ts, inplace=True)

    scaler = None  # scaler_fit(StandardScale(), train_ds.ts)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    ts_ex2_modeler = {
        'trans-ex2': [TransformerEx2, {'label_window_size': train_ds.window_size // 2, 'd_model': 128,
                                       'num_heads': 8, 'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                       'dim_ff': 512, 'dropout_rate': 0.}],
        'informer-ex2': [InformerEx2, {'label_window_size': train_ds.window_size // 2, 'd_model': 512,
                                       'num_heads': 8, 'num_encoder_layers': 2, 'num_decoder_layers': 1,
                                       'dim_ff': 2048, 'dropout_rate': 0.05}],
        'autoformer-ex2': [AutoformerEx2, {'label_window_size': train_ds.window_size // 2, 'd_model': 256,
                                           'num_heads': 8, 'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                           'dim_ff': 1024, 'dropout_rate': 0., 'moving_avg': 7}],
        'fedformer-ex2': [FEDformerEx2, {'label_window_size': train_ds.window_size // 2, 'd_model': 256,
                                         'num_heads': 8, 'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                         'dim_ff': 1024, 'activation': 'relu', 'moving_avg': 7, 'dropout_rate': 0.,
                                         'version': 'fourier', 'mode_select': 'random', 'modes': 32}],
        'deeptime': [DeepTIMe, {'layer_size': 256, 'inr_layers': 3, 'n_fourier_feats': 128,
                                'scales': [1, 2, 4, 8], 'dropout_rate': 0.1}],
    }

    model_cls, user_args = ts_ex2_modeler['deeptime']

    common_ds_args = get_common_kwargs(model_cls.__init__, train_ds.__dict__)
    combined_args = {**common_ds_args, **user_args}
    model = model_cls(**combined_args)

    model = covert_weight_types(model, torch_float_type)
    print(get_model_info(model))

    model_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_weights, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)
    stopper = EarlyStop(patience=10, delta=0.01, mode='rel')

    criterion = MSE()
    evaluator = Evaluator(['MSE', 'MAE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler)
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
