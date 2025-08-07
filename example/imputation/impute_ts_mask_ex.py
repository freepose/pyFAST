#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on **multiple sources** incomplete (sparse) time series imputation using exogenous data.

    This is an example incomplete time series forecasting using exogenous data.

    (1) pKa prediction using protein embeddings,
        this is the problem of **mask** time series forecasting using exogenous data.

    (2) Ensemble / Distributed Time series Learning (ETL) for Sparse Time Series Imputation/Forecasting.
        Impute the missing values of the time series solely based on exogenous data.
"""

import os, logging
import torch
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import StandardScale, MinMaxScale, scaler_fit, scaler_transform
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_weight_types
from fast.model.mts import TimeSeriesRNN, Transformer
from fast.model.mts_fusion import ExogenousDataDrivenPlugin as ExDD

from dataset.manage_smx_datasets import prepare_smx_datasets


def ts_mask_ex():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'cpu'

    task_config = {'ts': 'univariate', 'ts_mask': True, 'use_ex': True, 'dynamic_padding': True}
    seq, stride, horizon = 128, 128, 1 - 128  # input window is the output window
    train_ds = prepare_smx_datasets(data_root, 'phmd_2d_549_train', seq, seq, horizon, stride, device=ds_device, **task_config)
    val_ds = prepare_smx_datasets(data_root, 'phmd_2d_549_val', seq, seq, horizon, stride, device=ds_device, **task_config)
    test_ds = prepare_smx_datasets(data_root, 'phmd_2d_549_test', seq, seq, horizon, stride, device=ds_device, **task_config)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    scaler = scaler_fit(StandardScale(), train_ds.ts + val_ds.ts, train_ds.ts_mask + val_ds.ts_mask)
    ex_scaler = scaler_fit(StandardScale(), train_ds.ex_ts + val_ds.ex_ts, None)

    modeler = {
        'exddm-lstm': [ExDD, {'ex_model_cls': TimeSeriesRNN,
                              'ex_model_args': {'rnn_cls': 'lstm', 'hidden_size': 128, 'num_layers': 1,
                                                'bidirectional': False, 'dropout_rate': 0.05,
                                                'decoder_way': 'mapping'}}],
        'exddm-trans': [ExDD, {'ex_model_cls': Transformer,
                               'ex_model_args': {'label_window_size': seq, 'd_model': 512, 'num_heads': 8,
                                                 'num_encoder_layers': 6, 'num_decoder_layers': 6,
                                                 'dim_ff': 2048, 'dropout_rate': 0., 'activation': 'gelu'}}],
    }

    plugin_cls, user_args = modeler['exddm-lstm']

    common_ds_args = get_common_kwargs(plugin_cls.__init__, train_ds.__dict__)
    combined_args = {**common_ds_args, **user_args}
    model = plugin_cls(**combined_args)

    logger = logging.getLogger()
    model = covert_weight_types(model, torch_float_type)
    logger.info(get_model_info(model))
    # logger.info(str(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel', verbose=False)

    criterion = MSE()
    evaluator = Evaluator(['MAE', 'RMSE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=None,  # stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)
    logger.info(str(trainer))

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True,
                verbose=2)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, 32, None, False, is_online=False)
        logger.info('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, 32, None, False, is_online=False)
        logger.info('val {}'.format(results))

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    initial_logger()

    ts_mask_ex()
