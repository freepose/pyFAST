#!/usr/bin/env python
# encoding: utf-8

"""

    Examples on incomplete (sparse) time series forecasting (ITSF).
    The incomplete (sparse) time series data is represented by a dense tensor and a mask (indicator) tensor.

    (1) Single-source single/multivariate time series forecasting with missing values.

    (2) Multi-source single/multivariate time series forecasting with missing values.

    (3) Global static mask and training dynamic mask.


    Some tips for benefiting from the codes:

    (1) For incomplete time series datasets, this supports models working on both fixed-length and
        vary-length input windows.

        The codes avoid the two problems:
        (1.a) Relative time steps/points mis-alignment, such as irregular time series data.
        (1.b) Randomly dynamic padding on collected vary-length slicing windows, such as irregular time series data.

    (2) Computation efficiency: ITSF can work well on personal computers (PC).
        If dataset is large, it is recommended to load on main memory (RAM).
        If model is large, it is recommended to train on GPU(s).
        If both dataset and model are large, the datasets can be loaded on main memory (RAM) and the model
         can be trained on GPU(s).

"""

import logging, os

import torch
import torch.optim as optim

from fast import initial_seed, initial_logger, get_device, get_common_kwargs
from fast.data import StandardScale, MinMaxScale, scaler_fit, scaler_transform
from fast.data import RandomMasker, BlockMasker, VariableMasker, masker_generate
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE

from fast.model.base import get_model_info, covert_weight_types
from fast.model.mts import GAR, AR, VAR, ANN
from fast.model.mts import DLinear, NLinear, RLinear, STD, PatchMLP
from fast.model.mts import CNNRNN, CNNRNNRes
from fast.model.mts import Transformer, TSMixer, Timer
from fast.model.mts import COAT, TCOAT, CoDR, CTRL

from dataset.manage_sst_datasets import prepare_sst_datasets
from dataset.manage_smx_datasets import prepare_smx_datasets


def ts_mask():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'mps'

    """
        Sparse long-sequence time series forecasting problems: sparse decomposition, shapelet representation
    """
    task_config = {'ts': 'multivariate', 'ts_mask': True}
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'ETTh1', 48, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'SuzhouIPL_Sparse', 48, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'SDWPF_Sparse', 24 * 6, 6 * 6, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'WSTD2_Sparse', 7 * 24, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    train_ds, val_ds, test_ds = prepare_smx_datasets(data_root, 'PhysioNet', 1440, 1440, 1, 1, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smx_datasets(data_root, 'HumanActivity', 3000, 1000, 1, 1000, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smx_datasets(data_root, 'USHCN', 745, 31, 1, 31, (0.6, 0.2, 0.2), 'inter', ds_device, **task_config)

    """
        Global **static mask**
    """
    train_ds.ts_mask = masker_generate(RandomMasker(0.8), train_ds.ts_mask) # BlockMasker(12, 0.8) | VariableMasker(0.8)

    """
        Overwritable scalers and dynamic scalers. 
    """
    # overwrite_scaler = scaler_fit(MinMaxScale(), train_ds.ts, train_ds.ts_mask)
    # train_ds.ts = scaler_transform(overwrite_scaler, train_ds.ts, train_ds.ts_mask)
    # if val_ds is not None:
    #     val_ds.ts = scaler_transform(overwrite_scaler, val_ds.ts, val_ds.ts_mask)
    # if test_ds is not None:
    #     test_ds.ts = scaler_transform(overwrite_scaler, test_ds.ts, test_ds.ts_mask)

    # Dynamic scaling while training or evaluation.
    scaler = scaler_fit(StandardScale(), train_ds.ts, train_ds.ts_mask)

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    ts_modeler = {
        'gar': [GAR, {'activation': 'relu'}],
        'ar': [AR, {'activation': 'relu'}],
        'var': [VAR, {'activation': 'linear'}],
        'ann': [ANN, {'hidden_sizes': [256] * 10, 'layer_norm': 'LN', 'activation': 'linear'}],
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
                        "activation": 'linear', "use_instance_scale": True, "dropout_rate": 0.05}]
    }

    model_cls, user_args = ts_modeler['coat']

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
    stopper = EarlyStop(patience=5, delta=0.01, mode='rel', verbose=False)
    dynamic_mask = BlockMasker(12, 0.9) # RandomMask(0.2) ｜ VariableMask(0.2) | BlockMask(12, 0.2)

    criterion = MSE()
    evaluator = Evaluator(['MSE', 'MAE'])

    trainer = Trainer(get_device(model_device), model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, # stopper=stopper,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler)
    loger.info(str(trainer))

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True, forecast_mask=dynamic_mask,
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
