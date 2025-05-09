#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on single source univariate / multivariate time series forecasting.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_device, get_common_params
from fast.data import AbstractScale, MinMaxScale
from fast.train import Trainer
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_parameters
from experiment.modeler.ts import ts_modeler

from dataset.prepare_xmcdc import load_xmcdc_sst
from dataset.prepare_industrial_power_load import load_industrial_power_load_sst as load_ipl_sst


def main():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    device = get_device('cpu')

    ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    (train_ds, val_ds), (scaler, ex_scaler) = load_xmcdc_sst(freq='1day', **ds_params)

    # ds_params = {'input_window_size': 8 * 24, 'output_window_size': 24, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_ipl_sst(data_root, interpolate_type='li', **ds_params)

    model_cls, user_settings = ts_modeler['ar']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print('{}\n{}'.format(train_ds, val_ds))

    model = covert_parameters(model, torch_float_type)
    print(get_model_info(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.996)

    criterion = MSE()
    evaluator = Evaluator(['MAE', 'RMSE', 'PCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=False,
                verbose=2)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    main()
