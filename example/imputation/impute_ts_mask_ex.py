#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on **multiple sources** incomplete (sparse_fusion) time series imputation using (dense) exogenous data.

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
from fast.model.mts import GAR, AR, VAR, ANN, DLinear, NLinear, RLinear, STD, PatchMLP, CNNRNN, CNNRNNRes
from fast.model.mts import TimeSeriesRNN, Transformer, TSMixer
from fast.model.mts import COAT, TCOAT, CoDR, CTRL
from fast.model.mts_fusion import ExogenousDataDrivenPlugin as ExDD

from dataset.manage_smt_datasets import prepare_smt_datasets
from dataset.manage_smi_datasets import prepare_smi_datasets


def ts_mask_ex():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'cpu'

    task_config = {'ts': 'univariate', 'ts_mask': True, 'use_ex': True, 'dynamic_padding': True}
    task_config.update(**{'window_size': 256, 'stride': 128})
    train_ds = prepare_smi_datasets(data_root, 'phmd_2d_549_train', device=ds_device, **task_config)
    val_ds = prepare_smi_datasets(data_root, 'phmd_2d_549_val', device=ds_device, **task_config)
    test_ds = prepare_smi_datasets(data_root, 'phmd_2d_549_test', device=ds_device, **task_config)

    # window_size = 10
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'DIgSILENT', window_size, window_size, 1 - window_size, 1, (0.7, 0.1, 0.2), 'inter', ds_device, **task_config)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    scaler = scaler_fit(StandardScale(), train_ds.ts + val_ds.ts, train_ds.ts_mask_input + val_ds.ts_mask_input)
    ex_scaler = scaler_fit(StandardScale(), train_ds.ex_ts + val_ds.ex_ts, None)

    modeler = {
        'exddm-ar': [ExDD, {'ex_model_cls': AR, 'ex_model_args': {'activation': 'linear'}}],
        'exddm-gar': [ExDD, {'ex_model_cls': GAR, 'ex_model_args': {'activation': 'linear'}}],
        'exddm-var': [ExDD, {'ex_model_cls': VAR, 'ex_model_args': {'activation': 'relu'}}],
        'exddm-ann': [ExDD, {'ex_model_cls': ANN,
                             'ex_model_args': {'layer_norm': 'LN', 'hidden_sizes': [64], 'activation': 'linear'}}],
        'exddm-nlinear': [ExDD, {'ex_model_cls': NLinear, 'ex_model_args': {'mapping': 'ar'}}],
        'exddm-dlinear': [ExDD, {'ex_model_cls': DLinear, 'ex_model_args': {'kernel_size': 7}}],
        'exddm-rlinear': [ExDD, {'ex_model_cls': RLinear,
                                 'ex_model_args': {'mapping': 'ar', 'd_model': 1024, 'use_instance_scale': True}}],
        'exddm-std': [ExDD, {'ex_model_cls': STD,
                             'ex_model_args': {'kernel_size': 19, 'd_model': 128, 'use_instance_scale': True}}],
        'exddm-patchmlp': [ExDD, {'ex_model_cls': PatchMLP,
                                  'ex_model_args': {'d_model': 256, 'use_instance_scale': True, 'num_encoder_layers': 1,
                                                    'patch_lens': [64, 32]}}],
        'exddm-cnnrnn': [ExDD, {'ex_model_cls': CNNRNN,
                                'ex_model_args': {'cnn_out_channels': 64, 'cnn_kernel_size': 5, 'rnn_cls': 'gru',
                                                  'rnn_hidden_size': 32, 'rnn_num_layers': 1, 'rnn_bidirectional': True,
                                                  'dropout_rate': 0.0, 'decoder_way': 'mapping'}}],
        'exddm-cnnrnnres': [ExDD, {'ex_model_cls': CNNRNNRes,
                                   'ex_model_args': {'cnn_out_channels': 64, 'cnn_kernel_size': 5, 'rnn_cls': 'gru',
                                                     'rnn_hidden_size': 32, 'rnn_num_layers': 1,
                                                     'rnn_bidirectional': True, 'dropout_rate': 0.0,
                                                     'decoder_way': 'mapping', 'residual_window_size': 1,
                                                     'residual_ratio': 0.9}}],
        'exddm-transformer': [ExDD, {'ex_model_cls': Transformer,
                                     'ex_model_args': {'label_window_size': 128, 'd_model': 1024, 'num_heads': 8,
                                                       'num_encoder_layers': 2, 'num_decoder_layers': 1, 'dim_ff': 4096,
                                                       'dropout_rate': 0.05}}],
        'exddm-tsmixer': [ExDD, {'ex_model_cls': TSMixer,
                                 'ex_model_args': {'num_blocks': 1, 'block_hidden_size': 128, 'dropout_rate': 0.05,
                                                   'use_instance_scale': True}}],
        'exddm-coat': [ExDD, {'ex_model_cls': COAT,
                              'ex_model_args': {'mode': 'sa', 'activation': 'relu', 'use_instance_scale': False,
                                                'dropout_rate': 0.05}}],
        'exddm-tcoat': [ExDD, {'ex_model_cls': TCOAT,
                               'ex_model_args': {'rnn_hidden_size': 64, 'rnn_num_layers': 2, 'rnn_bidirectional': False,
                                                 'residual_window_size': 16, 'residual_ratio': 0.9,
                                                 'dropout_rate': 0.05}}],
        'exddm-codr': [ExDD, {'ex_model_cls': CoDR, 'ex_model_args': {'horizon': -127, 'hidden_size': 32,
                                                                      'use_window_fluctuation_extraction': True,
                                                                      'dropout_rate': 0.05}}],
        'exddm-ctrl': [ExDD, {'ex_model_cls': CTRL,
                              'ex_model_args': {'rnn_hidden_size': 8, 'rnn_num_layers': 1, 'rnn_bidirectional': True,
                                                'activation': 'linear', 'use_instance_scale': True,
                                                'dropout_rate': 0.0}}]
    }

    plugin_cls, user_args = modeler['exddm-std']

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
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel')

    criterion = MSE()
    evaluator = Evaluator(['MAE', 'RMSE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=None,  # stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)
    logger.info(str(trainer))

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 200), batch_size=32, shuffle=True,
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
