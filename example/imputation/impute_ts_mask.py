#!/usr/bin/env python
# encoding: utf-8

"""

    Examples on time series imputation using mask modeling.

    The major settings of time series imputation on the datasets are as follows:
    (1) Dataset configuration: the output window is the output window (input_window_size == output_window_size),
                            horizon=1-window_size, stride = window_size.

    (2) Dynamic mask: the dynamic mask is applied on the input window, rather than the output window.
                      Use ``impute_mask`` to control the dynamic mask in training and evaluation.


"""

import logging, os

import torch
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import StandardScale, MinMaxScale, scaler_fit, scaler_transform, SMIDataset, SMTDataset
from fast.data import RandomMasker, BlockMasker, VariableMasker, masker_generate
from fast.data.mask import masker_generate_for_imputation
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_weight_types
from fast.model.mts import GAR, AR, VAR, ANN
from fast.model.mts import DLinear, NLinear, RLinear, STD, PatchMLP
from fast.model.mts import CNNRNN, CNNRNNRes
from fast.model.mts import Transformer, TSMixer, Timer
from fast.model.mts import COAT, TCOAT, CoDR, CTRL

from fast.impute import GRUD, RITS, RITSI

from dataset.manage_smi_datasets import prepare_smi_datasets


def ts_mask():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'mps'

    task_config = {'ts': 'multivariate', 'bidirectional': False, 'dynamic_padding': False}

    train_ds, val_ds, test_ds = prepare_smi_datasets(data_root, 'ETTh1', 48, 48, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smi_datasets(data_root, 'PeMS03', 48, 48, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smi_datasets(data_root, 'PhysioNet-simple', 2880, 2880, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smi_datasets(data_root, 'USHCN', 1490, 1490, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)

    """
        Global **static mask**. This simulates the missing mechanism of real world.
        chose: RandomMasker(0.8) | BlockMasker(12, 0.8) | VariableMasker(0.8)
    """
    masker_generate_for_imputation(RandomMasker(0.9), train_ds.ts_mask_input, train_ds.ts_mask_output, inplace=True)
    if val_ds is not None:
        masker_generate_for_imputation(RandomMasker(0.9), val_ds.ts_mask_input, val_ds.ts_mask_output, inplace=True)
    if test_ds is not None:
        masker_generate_for_imputation(RandomMasker(0.9), test_ds.ts_mask_input, test_ds.ts_mask_output, inplace=True)

    # dynamic_masker = RandomMasker(0.85)

    """
        Overwritable (static) scalers and dynamic scalers. 
    """
    overwrite_scaler = scaler_fit(MinMaxScale(), train_ds.ts, train_ds.ts_mask_input)
    scaler_transform(overwrite_scaler, train_ds.ts, train_ds.ts_mask_input, inplace=True)
    if val_ds is not None:
        scaler_transform(overwrite_scaler, val_ds.ts, val_ds.ts_mask_input, inplace=True)
    if test_ds is not None:
        scaler_transform(overwrite_scaler, test_ds.ts, test_ds.ts_mask_input, inplace=True)

    # Dynamic scaling while training or evaluation.
    dynamic_scaler = None # scaler_fit(StandardScale(), train_ds.ts, train_ds.ts_mask)

    # Reindex the dataset to calculate densities.
    for ds in [train_ds, val_ds, test_ds]:
        if isinstance(ds, (SMTDataset, SMIDataset)):
            ds.index_dataset()
    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    ts_modeler = {
        'gar': [GAR, {'activation': 'relu'}],
        'ar': [AR, {'activation': 'relu'}],
        'var': [VAR, {'activation': 'linear'}],
        'ann': [ANN, {'hidden_sizes': [train_ds.window_size] * 3, 'layer_norm': None, 'activation': 'linear'}],
        'cnnrnn': [CNNRNN, {'cnn_out_channels': 50, 'cnn_kernel_size': 9,
                            'rnn_cls': 'gru', 'rnn_hidden_size': 32, 'rnn_num_layers': 1,
                            'rnn_bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
        'cnnrnnres': [CNNRNNRes, {'cnn_out_channels': 50, 'cnn_kernel_size': 9,
                                  'rnn_cls': 'gru', 'rnn_hidden_size': 32, 'rnn_num_layers': 1,
                                  'rnn_bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping',
                                  'residual_window_size': 5, 'residual_ratio': 0.1}],
        'nlinear': [NLinear, {'mapping': 'gar'}],
        'dlinear': [DLinear, {'kernel_size': 135, 'mapping': 'gar'}],
        'rlinear': [RLinear, {'dropout_rate': 0., 'use_instance_scale': True, 'mapping': 'gar',
                              'd_model': 128}],  # AAAI 2025
        'std': [STD, {'kernel_size': 75, 'd_model': 512, 'use_instance_scale': True}],
        'patchmlp': [PatchMLP, {'kernel_size': 13, 'd_model': 512, 'patch_lens': [256, 128, 96, 48],
                                'num_encoder_layers': 1, 'use_instance_scale': True}],  # AAAI 2025
        'transformer': [Transformer, {'label_window_size': train_ds.output_window_size, 'd_model': 512, 'num_heads': 8,
                                      'num_encoder_layers': 1, 'num_decoder_layers': 1, 'dim_ff': 2048,
                                      'dropout_rate': 0.05}],
        'timer': [Timer, {'patch_len': 4, 'd_model': 64, 'num_heads': 8, 'e_layers': 1, 'dim_ff': 512,
                          'activation': 'relu', 'dropout_rate': 0.}],
        'tsmixer': [TSMixer, {'num_blocks': 2, 'block_hidden_size': 2048, 'dropout_rate': 0.05,
                              'use_instance_scale': True}],  # TMLR 2023
        'coat': [COAT, {'mode': 'dr', 'activation': 'linear', 'use_instance_scale': False, 'dropout_rate': 0.}],
        "tcoat": [TCOAT, {"rnn_hidden_size": 8, "rnn_num_layers": 2, "rnn_bidirectional": True,
                          "residual_window_size": 240, "residual_ratio": 0.5, "dropout_rate": 0.05}],
        "codr": [CoDR, {"horizon": 1, "hidden_size": 179,
                        "use_window_fluctuation_extraction": True, "dropout_rate": 0.05}],
        "ctrl": [CTRL, {"rnn_hidden_size": 32, "rnn_num_layers": 2, "rnn_bidirectional": False,
                        "activation": 'linear', "use_instance_scale": True, "dropout_rate": 0.05}],

        # imputation models
        "gru_d": [GRUD, {'hidden_size': 64}],
        "rits": [RITS, {'hidden_size': 64}],
        "rits-i": [RITSI, {'hidden_size': 64}],
    }

    model_cls, user_args = ts_modeler['rits-i']

    common_ds_args = get_common_kwargs(model_cls.__init__, train_ds.__dict__)
    combined_args = {**common_ds_args, **user_args}
    model = model_cls(**combined_args)

    loger = logging.getLogger()
    model = covert_weight_types(model, torch_float_type)
    loger.info(get_model_info(model))
    # loger.info(str(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.0001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.996)
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel')

    criterion = MSE()
    evaluator = Evaluator(['MSE', 'MAE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=dynamic_scaler)
    loger.info(str(trainer))

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True, # impute_masker=dynamic_masker,
                verbose=2)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, 32, None, False, is_online=False)
        loger.info('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, 32, None, False, is_online=False)
        loger.info('val {}'.format(results))

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    initial_logger()

    ts_mask()
