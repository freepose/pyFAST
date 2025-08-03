#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on single-source / multi-source univariate / multivariate time series forecasting.

    (1) Single-source univariate time series forecasting, such as ETTh1, ETTh2, ETTm1, ETTm2, etc.

    (2) Single-source multivariate time series forecasting, such as ETTh1, Traffic, Electricity, etc.

    (3) Multi-source univariate time series forecasting, such as SH_diabetes, SDWPF, GreeceWPF, etc.

        This supports for multi-resolution time series forecasting.
        Multi-resolution means that several time series with several frequencies (i.e., time intervals).

        Infinite time granularity is supported, such as 1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour, etc.

    (4) Multi-source multivariate time series forecasting. None examples are provided yet.

    Some tips for benefiting from the codes:

    (1) Device: the dataset device and model device can be the same or different, and they work well together.
        If they are different, the dataset will be moved to the model device before training.
        If the dataset is large, it is recommended to set the dataset device to 'cpu' and
            the model device to 'cuda' or 'mps'.

    (2) Normalization: the scaler on target variables is fitted on the training set (maybe add validation set),
        and then applied to transform the validation and test sets.

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
from fast.model.mts import GAR, AR, VAR, ANN, TimeSeriesRNN, EncoderDecoder
from fast.model.mts import NHiTS, DLinear, NLinear, RLinear, STD, Amplifier, PatchMLP
from fast.model.mts import TemporalConvNet, CNNRNN, CNNRNNRes, LSTNet
from fast.model.mts import DeepResidualNetwork
from fast.model.mts import Transformer, Informer, Autoformer, FiLM, Triformer, FEDformer, Crossformer
from fast.model.mts import TimesNet, PatchTST, STAEformer, iTransformer, TSMixer, TimeXer, TimeMixer
from fast.model.mts import TimesFM, Timer, TSLANet
from fast.model.mts import STID, STNorm, MAGNet, GraphWaveNet, FourierGNN
from fast.model.mts import COAT, GAIN

from dataset.prepare_xmcdc import load_xmcdc_sst
from dataset.manage_sst_datasets import prepare_sst_datasets, verify_sst_datasets
from dataset.manage_smt_datasets import prepare_smt_datasets


def main():
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'
    torch_float_type = torch.float32
    ds_device, model_device = 'cpu', 'mps'

    task_config = {'ts': 'univariate'}
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'XMCDC_1day', 10, 1, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'XMCDC_1week', 10, 1, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'ETTh1', 48, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'ExchangeRate_x1000', 4 * 7, 7, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'SuzhouIPL', 48, 24, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_sst_datasets(data_root, 'TurkeyWPF', 6 * 24, 6 * 6, 1, 1, (0.7, 0.1, 0.2), ds_device, **task_config)

    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'GreeceWPF', 10 * 24, 1 * 24, 1, 1, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'SDWPF', 6 * 24, 6 * 6, 1, 1, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'WSTD2', 6 * 24, 6 * 6, 1, 1, (0.7, 0.1, 0.2), 'intra', ds_device, **task_config)
    # train_ds, val_ds, test_ds = prepare_smt_datasets(data_root, 'SH_diabetes', 6 * 4, 2, 1, 1, (0.7, 0.1, 0.2), 'inter', ds_device, **task_config)

    scaler = scaler_fit(StandardScale(), train_ds.ts)
    train_ds.ts = scaler_transform(scaler, train_ds.ts)
    if val_ds is not None:
        val_ds.ts = scaler_transform(scaler, val_ds.ts)
    if test_ds is not None:
        test_ds.ts = scaler_transform(scaler, test_ds.ts)
    scaler = None

    print('\n'.join([str(ds) for ds in [train_ds, val_ds, test_ds]]))

    ts_modeler = {
        'gar': [GAR, {'activation': 'linear'}],
        'ar': [AR, {'activation': 'linear'}],
        'var': [VAR, {'activation': 'linear'}],
        'ann': [ANN, {'hidden_size': 32}],
        'drn': [DeepResidualNetwork, {'hidden_size': 64, 'number_stacks': 7, 'number_blocks_per_stack': 1,
                                      'use_rnn': False}],
        'rnn': [TimeSeriesRNN, {'rnn_cls': 'gru', 'hidden_size': 32, 'num_layers': 1,
                                'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
        'ed': [EncoderDecoder, {'rnn_cls': 'gru', 'hidden_size': 32, 'num_layers': 1,
                                'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
        'tcn': [TemporalConvNet, {'num_channels': [32], 'kernel_size': 2, 'dropout_rate': 0.2}],
        'cnnrnn': [CNNRNN, {'cnn_out_channels': 50, 'cnn_kernel_size': 9,
                            'rnn_cls': 'gru', 'rnn_hidden_size': 32, 'rnn_num_layers': 1,
                            'rnn_bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
        'cnnrnnres': [CNNRNNRes, {'cnn_out_channels': 50, 'cnn_kernel_size': 9,
                                  'rnn_cls': 'gru', 'rnn_hidden_size': 32, 'rnn_num_layers': 1,
                                  'rnn_bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping',
                                  'residual_window_size': 5, 'residual_ratio': 0.1}],
        'lstnet': [LSTNet, {'cnn_out_channels': 50, 'cnn_kernel_size': 9,
                            'rnn_hidden_size': 50, 'rnn_num_layers': 1,
                            'skip_window_size': 24, 'skip_gru_hidden_size': 20,
                            'highway_window_size': 24, 'dropout_rate': 0.}],
        'nhits': [NHiTS, {'n_blocks': [1, 1, 1], 'n_layers': [2] * 8,
                          'hidden_size': [[512, 512], [512, 512], [512, 512]],
                          'pooling_sizes': [8, 8, 8], 'downsample_frequencies': [24, 12, 1],
                          'pooling_mode': 'max', 'interpolation_mode': 'linear',
                          'dropout': 0.0, 'activation': 'ReLU', 'initialization': 'lecun_normal',
                          'batch_normalization': False, 'shared_weights': False, 'naive_level': True}],
        'nlinear': [NLinear, {'mapping': 'gar'}],
        'dlinear': [DLinear, {'kernel_size': 25, 'mapping': 'gar'}],
        'rlinear': [RLinear, {'dropout_rate': 0., 'use_instance_scale': True, 'mapping': 'gar',
                              'd_model': 128}],  # AAAI 2025
        'std': [STD, {'kernel_size': 7, 'd_model': 128, 'use_instance_scale': True}],
        'amplifier': [Amplifier, {'kernel_size': 7, 'hidden_size': 128, 'use_sci_block': True,
                                  'use_instance_scale': True}],   # AAAI 2025
        'patchmlp': [PatchMLP, {'kernel_size': 13, 'd_model': 512, 'patch_lens': [48, 24, 12, 6],
                                'num_encoder_layers': 1, 'use_instance_scale': True}],   # AAAI 2025
        'transformer': [Transformer, {'label_window_size': 7, 'd_model': 128, 'num_heads': 8,
                                      'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                      'dim_ff': 512, 'dropout_rate': 0.}],
        'informer': [Informer, {'label_window_size': 48, 'd_model': 512, 'num_heads': 8,
                                'num_encoder_layers': 2, 'num_decoder_layers': 1,
                                'dim_ff': 2048, 'dropout_rate': 0.05}],
        'autoformer': [Autoformer, {'label_window_size': 0, 'd_model': 256, 'num_heads': 8,
                                    'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                    'dim_ff': 1024, 'dropout_rate': 0., 'moving_avg': 7}],
        'fedformer': [FEDformer, {'label_window_size': 4, 'd_model': 256, 'num_heads': 8,
                                  'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                  'dim_ff': 1024, 'activation': 'relu', 'moving_avg': 7, 'dropout_rate': 0.,
                                  'version': 'fourier', 'mode_select': 'random', 'modes': 32}],
        'film': [FiLM, {'d_model': 128, 'use_instance_scale': True}],
        'triformer': [Triformer, {'channels': 32, 'patch_sizes': [8, 3], 'mem_dim': 5}],
        'crossformer': [Crossformer, {'d_model': 128, 'num_heads': 8, 'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                      'dim_ff': 512, 'dropout_rate': 0., 'factor': 5,
                                      'seg_len': 24, 'win_size': 2}],
        'timesnet': [TimesNet, {'d_model': 512, 'num_encoder_layers': 1, 'dim_ff': 2048, 'dropout_rate': 0.,
                                'num_kernels': 6, 'top_k': 5, 'use_instance_scale': False}],
        'patchtst': [PatchTST, {'d_model': 64, 'num_heads': 8, 'num_encoder_layers': 6, 'dim_ff': 256,
                                'patch_len': 5, 'patch_stride': 2, 'patch_padding': 2,
                                'use_instance_scale': True}],
        'staeformer': [STAEformer, {'input_dim': 1, 'output_dim': 1, 'input_embedding_dim': 24,
                                    'tod_steps_per_day': 24, 'tod_embedding_dim': 0, 'dow_embedding_dim': 0,
                                    'spatial_embedding_dim': 0, 'adapti。。ve_embedding_dim': 80,
                                    'dim_ff': 256, 'num_heads': 4, 'num_layers': 3,
                                    'dropout_rate': 0.1, 'use_mixed_proj': True}],
        'itransformer': [iTransformer, {'d_model': 128, 'num_heads': 2, 'num_encoder_layers': 6, 'dim_ff': 512,
                                        'activation': 'relu', 'dropout_rate': 0., 'use_instance_scale': True}],
        'timesfm': [TimesFM, {'d_model': 64, 'num_heads': 2, 'num_layers': 2,
                              'dim_ff': 256, 'dropout_rate': 0.}],
        'timer': [Timer, {'patch_len': 4, 'd_model': 64, 'num_heads': 8, 'e_layers': 1, 'dim_ff': 512,
                          'activation': 'relu', 'dropout_rate': 0.}],
        'tsmixer': [TSMixer, {'num_blocks': 2, 'block_hidden_size': 2048, 'dropout_rate': 0.05,
                              'use_instance_scale': True}],  # TMLR 2023
        'timexer': [TimeXer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 2,
                              'dim_ff': 2048, 'dropout_rate': 0.05, 'activation': 'relu',
                              'patch_len': 16, 'use_instance_scale': True}],
        'timemixer': [TimeMixer, {'d_model': 16, 'num_heads': 8, 'num_encoder_layers': 1,
                                  'dim_ff': 64, 'dropout_rate': 0.05, 'moving_avg': 25,
                                  'top_k': 5, 'channel_independence': True, 'decomposition_method': 'moving_avg',
                                  'down_sampling_method': 'avg', 'down_sampling_window': 1, 'down_sampling_layers': 1,
                                  'use_instance_scale': True}],
        'tslanet': [TSLANet, {'patch_len': train_ds.input_window_size // 4, 'patch_stride': None, 'embedding_dim': 64,
                              'mlp_hidden_size': None, 'num_blocks': 3, 'block_type': 'asb_icb',
                              'dropout_rate': 0.5, 'use_instance_scale': True}],  # ICML 2024
        'stid': [STID, {'node_dim': 32, 'embed_dim': 1024, 'input_dim': 1, 'num_layer': 1, 'if_node': True}],
        'stnorm': [STNorm, {'tnorm_bool': True, 'snorm_bool': True,
                            'channels': 16, 'kernel_size': 2, 'blocks': 1, 'layers': 2}],
        'magnet': [MAGNet, {'label_window_size': train_ds.input_window_size, 'conv2d_in_channels': 1,
                            'residual_channels': 32,  'conv_channels': 32, 'skip_channels': 4, 'end_channels': 128,
                            'node_dim': 40, 'tanhalpha': 3.0, 'static_feat': None, 'dilation_exponential': 1,
                            'kernel_size': 7, 'gcn_depth': 2, 'gcn_true': False, 'propalpha': 0.05,
                            'layer_norm_affline': True, 'buildA_true': True, 'predefined_A': None, 'dropout': 0.3}],
        'gwn': [GraphWaveNet, {'out_dim': 1, 'supports': None, 'gcn_bool': True, 'addaptadj': True,
                               'aptinit': None, 'in_dim': 1, 'residual_channels': 32, 'dilation_channels': 32,
                               'skip_channels': 256, 'end_channels': 512, 'kernel_size': 2,
                               'blocks': 1, 'layers': 2, 'dropout': 0.3}],
        'fgnn': [FourierGNN, {'embed_size': 128, 'hidden_size': 256, 'hard_thresholding_fraction': 1,
                              'hidden_size_factor': 1, 'sparsity_threshold': 0.01}],
        'coat': [COAT, {'mode': 'dr', 'activation': 'linear', 'use_instance_scale': False, 'dropout_rate': 0.}],
        'gain': [GAIN, {'gat_hidden_size': 64, 'gat_nhead': 128, 'gru_hidden_size': 8, 'gru_num_layers': 1,
                        'cnn_kernel_size': 3, 'cnn_out_channels': 16, 'highway_window_size': 10, 'dropout_rate': 0.5}],
    }

    model_cls, user_settings = ts_modeler['coat']

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
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=stopper,
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
    # verify_sst_datasets()