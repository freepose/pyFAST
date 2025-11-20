#!/usr/bin/env python
# encoding: utf-8

"""

    Examples on incomplete irregular time series forecasting (IrTSF).

"""

import logging, os

import torch
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import StandardScale, MinMaxScale, scaler_fit, scaler_transform
from fast.data import RandomMasker, BlockMasker, VariableMasker, masker_generate
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MAE, MSE

from fast.model.base import get_model_info, covert_weight_types
from fast.model.mts import GAR, AR, VAR, TimeSeriesRNN, EncoderDecoder, Transformer

from fast.data.smir_dataset import smir_collate_fn
from fast.data.smir_dataset import ThresholdSupervisedStrategy as Threshold
from fast.data.smir_dataset import PairwiseSupervisedStrategy as Pairwise
from fast.data.smir_dataset import WindowSupervisedStrategy as Window

from dataset.manage_smir_datasets import prepare_smir_datasets


def irregular_ts_mask():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'G:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'mps' if os.name == 'posix' else 'cuda:0'

    task_config = {'ts': 'multivariate', 'use_ex': False, 'device': ds_device, 'shuffle': True}

    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'PhysioNet-ir', Window(1, 1, 1, 1), (0.6, 0.2, 0.2), **task_config)

    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'PhysioNet-ir', Threshold(1440), (0.6, 0.2, 0.2), **task_config)
    train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'MIMIC-III-Ext-tPatchGNN-ir', Threshold(1440), (0.6, 0.2, 0.2), **task_config)
    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'MIMIC-III-v1.4-ir', Threshold(1440), (0.6, 0.2, 0.2), **task_config)
    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'MIMIC-IV-v3.1-ir', Threshold(1440), (0.6, 0.2, 0.2), **task_config)

    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'HR-VILAGE-3K3M-ir-pca', Threshold(1), (0.6, 0.2, 0.2), **task_config)
    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'HR-VILAGE-3K3M-ir-transpose', Threshold(1), (0.06, 0.02, 0.02), transpose=True, **task_config)
    # train_ds, val_ds, test_ds = prepare_smir_datasets(data_root, 'HR-VILAGE-3K3M-ir-transpose', Pairwise(False), (0.6, 0.2, 0.2), transpose=True, **task_config)

    """
        Global **static mask**. This simulates the missing mechanism of real world.
        chose: RandomMasker(0.8) | BlockMasker(12, 0.8) | VariableMasker(0.8)
    """

    # masker_generate(RandomMasker(0.8), train_ds.ts_mask, inplace=True)
    # if val_ds is not None:
    #     masker_generate(RandomMasker(0.8), val_ds.ts_mask, inplace=True)
    # if test_ds is not None:
    #     masker_generate(RandomMasker(0.8), test_ds.ts_mask, inplace=True)
    #
    # dynamic_masker = RandomMasker(0.95)

    """
        Overwritable scalers and dynamic scalers. 
    """
    overwrite_scaler = scaler_fit(MinMaxScale(), train_ds.ts, train_ds.ts_mask)
    scaler_transform(overwrite_scaler, train_ds.ts, train_ds.ts_mask, inplace=True, show_progress=True)
    if val_ds is not None:
        scaler_transform(overwrite_scaler, val_ds.ts, val_ds.ts_mask, inplace=True, show_progress=True)
    if test_ds is not None:
        scaler_transform(overwrite_scaler, test_ds.ts, test_ds.ts_mask, inplace=True, show_progress=True)

    # Dynamic scaling while training or evaluation.
    scaler = None  # scaler_fit(StandardScale(), train_ds.ts, train_ds.ts_mask)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    """
        Irregular models should support mask and varying time steps.
    """

    ir_ts_mask_modeler = {
        'gar': [GAR, {}],
        'ar': [AR, {}],
        'var': [VAR, {}],

        'rnn': [TimeSeriesRNN, {
            'rnn_cls': 'rnn', 'hidden_size': 64, 'num_layers': 1,
            'bidirectional': False, 'dropout_rate': 0.05,
            'decoder_way': 'inference',
        }],

        'gru': [TimeSeriesRNN, {
            'rnn_cls': 'gru', 'hidden_size': 64, 'num_layers': 1,
            'bidirectional': False, 'dropout_rate': 0.05,
            'decoder_way': 'inference',
        }],

        'lstm': [TimeSeriesRNN, {
            'rnn_cls': 'lstm', 'hidden_size': 64, 'num_layers': 1,
            'bidirectional': False, 'dropout_rate': 0.05,
            'decoder_way': 'inference',
        }],

        'ed_rnn': [EncoderDecoder, {
            'rnn_cls': 'rnn', 'hidden_size': 64, 'num_layers': 1,
            'bidirectional': False, 'dropout_rate': 0.05,
            'decoder_way': 'inference',
        }],

        'ed_gru': [EncoderDecoder, {
            'rnn_cls': 'gru', 'hidden_size': 64, 'num_layers': 1,
            'bidirectional': False, 'dropout_rate': 0.05,
            'decoder_way': 'inference',
        }],

        'ed_lstm': [EncoderDecoder, {
            'rnn_cls': 'lstm', 'hidden_size': 64, 'num_layers': 1,
            'bidirectional': False, 'dropout_rate': 0.05,
            'decoder_way': 'inference',
        }],

        'transformer': [Transformer, {
            'label_window_size': None, 'd_model': 512, 'num_heads': 8,
            'num_encoder_layers': 1, 'num_decoder_layers': 1, 'dim_ff': 2048,
            'dropout_rate': 0.05, "norm_first": False,
        }],

    }

    model_cls, user_args = ir_ts_mask_modeler['gru']

    common_ds_args = get_common_kwargs(model_cls.__init__, train_ds.__dict__)
    combined_args = {**common_ds_args, **user_args}
    model = model_cls(**combined_args)

    loger = logging.getLogger()
    model = covert_weight_types(model, torch_float_type)
    loger.info(get_model_info(model))
    # loger.info(str(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0005, weight_decay=0.00005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.996)
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel')
    criterion = MSE()
    evaluator = Evaluator(['MSE', 'MAE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler)
    loger.info(str(trainer))

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True, collate_fn=smir_collate_fn,
                verbose=2)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, 32, smir_collate_fn, False, is_online=False)
        loger.info('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, 32, smir_collate_fn, False, is_online=False)
        loger.info('val {}'.format(results))

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    initial_logger()

    irregular_ts_mask()
