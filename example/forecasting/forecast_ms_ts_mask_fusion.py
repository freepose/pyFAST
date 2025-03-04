#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on multiple sources univariate / multivariate incomplete time series forecasting (ITSF).

    This is an example incomplete time series forecasting using exogenous data.

    (1) pKa prediction using protein embeddings,
        this is the problem of **mask** time series forecasting using exogenous data.

"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_common_params
from fast.data import Scale, MinMaxScale, StandardScale
from fast.data import collate_fn
from fast.train import Trainer
from fast.metric import Evaluator, mean_absolute_scaled_error, mask_mean_squared_error

from fast.model.base import count_parameters, covert_parameters
from fast.model.mts_fusion import ARX, NARXMLP, NARXRNN
from fast.model.mts_fusion import DSAR, DGR, DGDR, MvT, GAINGE
from fast.model.mts_fusion import ExDDM

from example.prepare_pka import load_pka


def mask_mts_fusion():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seq_len, stride = 256, 128
    ds_params = {'input_window_size': seq_len, 'output_window_size': seq_len, 'horizon': 1 - seq_len, 'stride':  stride}
    train_ds, val_ds = load_pka(data_root, ds_params, 'phmd_2d_549', 'bdp')

    modeler = {
        'exdd': [ExDDM, {}],
    }

    model_cls, user_settings = modeler['exdd']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print('{}\n{}\n{}'.format(train_ds, val_ds, model))

    model_name = type(model).__name__
    model = covert_parameters(model, torch_float_type)
    print(model_name, count_parameters(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.996)

    criterion = mask_mean_squared_error  # nn.MSELoss()
    additive_criterion = getattr(model, 'loss', None)
    evaluator = Evaluator(['maskMAE', 'maskRMSE', 'maskPCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, additive_criterion=additive_criterion, evaluator=evaluator)

    trainer.fit(train_ds,
                val_ds,
                collate_fn=collate_fn,
                epoch_range=(1, 2000), batch_size=16, shuffle=True,
                verbose=True, display_interval=0)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(42)
    mask_mts_fusion()
