#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on **multiple sources** univariate / multivariate incomplete time series forecasting (ITSF).

    This is an example incomplete time series forecasting using exogenous data.

    (1) pKa prediction using protein embeddings,
        this is the problem of **mask** time series forecasting using exogenous data.

"""

import os
import torch
import torch.optim as optim

from fast import initial_seed, get_device, get_common_params
from fast.data import bdp_collate_fn
from fast.train import Trainer
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_parameters

from fast.model.mts import TimeSeriesRNN, Transformer
from fast.model.mts_fusion import ExogenousDataDrivenPlugin as ExDD

from dataset.prepare_pka import load_pka


def mask_mts_fusion():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    device = get_device('mps')

    seq_len, stride = 1, 1
    ds_params = {'input_window_size': seq_len, 'output_window_size': seq_len, 'horizon': 1 - seq_len, 'stride': stride}
    train_ds, val_ds = load_pka(data_root, ds_params, 'phmd_2d_549', 'bdp')

    modeler = {
        'exddm-lstm': [ExDD, {'ex_model_cls': TimeSeriesRNN,
                              'params': {'rnn_cls': 'lstm', 'hidden_size': 128, 'num_layers': 1,
                                         'bidirectional': False, 'dropout_rate': 0.05, 'decoder_way': 'mapping'}}],
        'exddm-trans': [ExDD, {'ex_model_cls': Transformer,
                               'params': {'label_window_size': 0, 'd_model': 512, 'num_heads': 8,
                                          'num_encoder_layers': 6, 'num_decoder_layers': 6,
                                          'dim_ff': 2048, 'dropout_rate': 0., 'activation': 'gelu'}}],
    }

    plugin, user_settings = modeler['exddm-lstm']

    common_ds_params = get_common_params(plugin.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = plugin(**model_settings)

    print('{}\n{}'.format(train_ds, val_ds))

    model = covert_parameters(model, torch_float_type)
    print(get_model_info(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)

    criterion = MSE()
    evaluator = Evaluator(['MAE', 'RMSE', 'PCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, evaluator=evaluator)

    trainer.fit(train_ds, val_ds,
                collate_fn=bdp_collate_fn,
                epoch_range=(1, 2000), batch_size=32, shuffle=True,
                verbose=2)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    mask_mts_fusion()
