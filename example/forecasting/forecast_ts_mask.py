#!/usr/bin/env python
# encoding: utf-8

"""

    Examples on incomplete (sparse) time series forecasting (ITSF).
    The incomplete (sparse) time series data is represented by a dense tensor and a mask (indicator) tensor.

    (1) Single-source single/multivariate time series forecasting with missing values.

    (2) Multi-source single/multivariate time series forecasting with missing values.

    Some tips for benefiting from the codes:

    (1) For irregular time series datasets, this supports models working on both fixed-length and
        vary-length input windows.

        The codes avoid the two problems:
        (1.a) Relative time steps/points mis-alignment.
        (1.b) Randomly dynamic padding on collected vary-length slicing windows.

    (2) Computation efficiency: ITSF can work well on personal computers (PC).

"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import StandardScale, MinMaxScale, scaler_fit, scaler_transform
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_parameters
from fast.model.mts import GAR, AR, VAR, ANN
from fast.model.mts import DLinear, NLinear, RLinear, STD
from fast.model.mts import Transformer

from dataset.manage_sst_datasets import prepare_sst_datasets
from dataset.manage_smt_datasets import prepare_smt_datasets


def main():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'cpu'

    task_config = {'ts': 'multivariate', 'ts_mask': True}
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'SuzhouIPL_Sparse', 48, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'SDWPF_Sparse', 24 * 6, 6 * 6, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'WSTD2_Sparse', 7 * 24, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'PhysioNet', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'HumanActivity', 3000, 1000, 1, 1000, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'USHCN', 745, 31, 1, 31, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)

    # fit scalers on training and validation datasets
    scaler = scaler_fit(MinMaxScale(), train_ds.ts, train_ds.ts_mask)
    train_ds.ts = scaler_transform(scaler, train_ds.ts, train_ds.ts_mask)
    if val_ds is not None:
        val_ds.ts = scaler_transform(scaler, val_ds.ts, val_ds.ts_mask)
    if test_ds is not None:
        test_ds.ts = scaler_transform(scaler, test_ds.ts, test_ds.ts_mask)
    scaler = None

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    ts_modeler = {
        'gar': [GAR, {'activation': 'linear'}],
        'ar': [AR, {'activation': 'relu'}],
        'var': [VAR, {'activation': 'linear'}],
        'ann': [ANN, {'hidden_size': 512}],
        'nlinear': [NLinear, {'mapping': 'gar'}],
        'dlinear': [DLinear, {'kernel_size': 75, 'mapping': 'gar'}],
        'rlinear': [RLinear, {'dropout_rate': 0., 'use_instance_scale': True, 'mapping': 'gar',
                              'd_model': 128}],  # AAAI 2025
        'std': [STD, {'kernel_size': 75, 'd_model': 512, 'use_instance_scale': True}],
        'transformer': [Transformer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 1,
                                      'num_decoder_layers': 1,  'dim_ff': 2048, 'dropout_rate': 0.}],
    }

    model_cls, user_settings = ts_modeler['std']

    common_ds_params = get_common_kwargs(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    model = covert_parameters(model, torch_float_type)
    print(get_model_info(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.996)
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel', verbose=False)

    criterion = MSE()
    evaluator = Evaluator(['MSE', 'MAE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=None, # stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler)
    print(trainer)

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True,
                verbose=2)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, 32, None, False, is_online=False)
        print('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, 32, None, False, is_online=False)
        print('val {}'.format(results))

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    initial_logger()
    main()
