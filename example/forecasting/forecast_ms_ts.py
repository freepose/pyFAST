#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on **multiple sources** time series forecasting.

    The time series length varies among the sources.
    In most occasions, the time series variables of all the sources is 1.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_device, get_common_params
from fast.train import Trainer
from fast.metric import Evaluator

from fast.model.base import count_parameters, covert_parameters
from experiment.modeler.ts import ts_modeler

from dataset.prepare_xmcdc import load_xmcdc_smt
from dataset.prepare_kdd2018_glucose import load_kdd2018_glucose_smt
from dataset.prepare_sh_diabetes import load_sh_diabetes_smt
from dataset.prepare_industrial_power_load import load_industrial_power_load_smt as load_ipl_smt
from dataset.prepare_greek_wind import load_greece_wpf_smt as load_gwpf_smt
from dataset.prepare_kdd2022_sdwpf import load_kdd2022_sdwpf_smt as load_sdwpf_smt


def main():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    device = get_device('cpu')

    stm_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    (train_ds, val_ds), (scaler, ex_scaler) = load_xmcdc_smt('1week', **stm_params)

    # (train_ds, val_ds), (scaler, ex_scaler) = load_kdd2018_glucose_smt(data_root, 0.8, 5 * 12, 6, 1, 1)
    # (train_ds, val_ds), (scaler, ex_scaler) = load_sh_diabetes_smt(data_root, 'all', None, False, 0.8, 6 * 4, 2, 1, 1)

    # ds_params = {'input_window_size': 6 * 24, 'output_window_size': 24, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_ipl_smt(data_root, interpolate_type='mice', **ds_params)

    # ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_gwpf_smt(data_root, '1day', None, False, 'inter', **ds_params, factor=0.001)

    # ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'stride': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_sdwpf_smt(data_root, '1day', None, False, 'inter', **ds_params, factor=0.001)

    model_cls, user_settings = ts_modeler['ar']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print(f'{train_ds}\n{val_ds}\n{model}')

    model_name = type(model).__name__
    model = covert_parameters(model, torch_float_type)
    print(model_name, count_parameters(model, 'M'))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0005, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)

    criterion = nn.MSELoss()
    additive_criterion = getattr(model, 'loss', None)
    evaluator = Evaluator(['MAE', 'RMSE', 'PCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, additive_criterion=additive_criterion, evaluator=evaluator,
                      global_scaler=scaler, global_ex_scaler=ex_scaler)

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True,
                verbose=True, display_interval=50)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    main()
