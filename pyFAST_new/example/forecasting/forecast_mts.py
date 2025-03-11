#!/usr/bin/env python
# encoding: utf-8

"""
    Examples on single source univariate / multivariate time series forecasting using exogenous time series.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim

from fast import initial_seed, get_common_params
from fast.data import Scale, MinMaxScale, StandardScale
from fast.train import Trainer
from fast.metric import Evaluator

from fast.model.base import count_parameters, covert_parameters
from fast.model.mts import GAR, AR, VAR, ANN, TimeSeriesRNN, EncoderDecoder
from fast.model.mts import NHiTS, DLinear, NLinear
from fast.model.mts import TemporalConvNet, CNNRNN, CNNRNNRes, LSTNet
from fast.model.mts import DeepResidualNetwork
from fast.model.mts import Transformer, Informer, Autoformer, FiLM, Triformer, FEDformer, Crossformer
from fast.model.mts import TimesNet, PatchTST, STAEformer, iTransformer, TimeXer, TimeMixer
from fast.model.mts import TimesFM, Timer
from fast.model.mts import STID, STNorm, MAGNet, GraphWaveNet, FourierGNN
from fast.model.mts import COAT, GAIN

from example.prepare_xmcdc import load_xmcdc_sts
from example.prepare_industrial_power_load import load_industrial_power_load_sts as load_ecpl_sts


def main():
    data_root = os.path.expanduser('~/data/') if os.name == 'posix' else 'D:/data/'
    torch_float_type = torch.float32
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    ds_params = {'input_window_size': 10, 'output_window_size': 1, 'horizon': 1, 'split_ratio': 0.8}
    (train_ds, val_ds), (scaler, ex_scaler) = load_xmcdc_sts('../dataset/xmcdc/', 'weekly', None, None, ds_params)

    # ds_params = {'input_window_size': 8 * 24, 'output_window_size': 24, 'horizon': 1, 'split_ratio': 0.8}
    # (train_ds, val_ds), (scaler, ex_scaler) = load_ecpl_sts(data_root, 'mice', None, None, ds_params, Scale())

    modeler = {
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
                            'skip_window_size': 24, 'skip_gru _hidden_size': 20,
                            'highway_window_size': 24, 'dropout_rate': 0.}],
        'nhits': [NHiTS, {'n_blocks': [1, 1, 1], 'n_layers': [2] * 8,
                          'hidden_size': [[512, 512], [512, 512], [512, 512]],
                          'pooling_sizes': [8, 8, 8], 'downsample_frequencies': [24, 12, 1],
                          'pooling_mode': 'max', 'interpolation_mode': 'linear',
                          'dropout': 0.0, 'activation': 'ReLU', 'initialization': 'lecun_normal',
                          'batch_normalization': False, 'shared_weights': False, 'naive_level': True}],
        'dlinear': [DLinear, {'individual': False, 'kernel_size': 7}],
        'nlinear': [NLinear, {'individual': False}],
        'transformer': [Transformer, {'label_window_size': 0, 'd_model': 128, 'num_heads': 8,
                                      'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                      'dim_ff': 512, 'dropout_rate': 0.}],
        'informer': [Informer, {'label_window_size': 0, 'd_model': 128, 'num_heads': 8,
                                'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                'dim_ff': 512, 'dropout_rate': 0.}],
        'autoformer': [Autoformer, {'label_window_size': 0, 'd_model': 256, 'num_heads': 8,
                                    'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                    'dim_ff': 1024, 'dropout_rate': 0., 'moving_avg': 7}],
        'fedformer': [FEDformer, {'label_window_size': 4, 'd_model': 256, 'num_heads': 8,
                                  'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                  'dim_ff': 1024, 'activation': 'relu', 'moving_avg': 7, 'dropout_rate': 0.,
                                  'version': 'fourier', 'mode_select': 'random', 'modes': 32}],
        'film': [FiLM, {'d_model': 128, 'use_instance_scale': True}],
        'triformer': [Triformer, {'channels': 32, 'patch_sizes': [5, 2], 'mem_dim': 5}],
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
                                    'spatial_embedding_dim': 0, 'adaptive_embedding_dim': 80,
                                    'dim_ff': 256, 'num_heads': 4, 'num_layers': 3,
                                    'dropout_rate': 0.1, 'use_mixed_proj': True}],
        'itransformer': [iTransformer, {'d_model': 128, 'num_heads': 2, 'num_encoder_layers': 6, 'dim_ff': 512,
                                        'activation': 'relu', 'dropout_rate': 0., 'use_instance_scale': True}],
        'timesfm': [TimesFM, {'d_model': 64, 'num_heads': 2, 'num_layers': 2,
                              'dim_ff': 256, 'dropout_rate': 0.}],
        'timer': [Timer, {'patch_len': 4, 'd_model': 64, 'num_heads': 8, 'e_layers': 1, 'dim_ff': 512,
                          'activation': 'relu', 'dropout_rate': 0.}],
        'timexer': [TimeXer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 2,
                              'dim_ff': 2048, 'dropout_rate': 0.05, 'activation': 'relu',
                              'patch_len': 16, 'use_instance_scale': True}],
        'timemixer': [TimeMixer, {'d_model': 16, 'num_heads': 8, 'num_encoder_layers': 1,
                                  'dim_ff': 64, 'dropout_rate': 0.05, 'moving_avg': 25,
                                  'top_k': 5, 'channel_independence': True, 'decomposition_method': 'moving_avg',
                                  'down_sampling_method': 'avg', 'down_sampling_window': 1, 'down_sampling_layers': 1,
                                  'use_instance_scale': True}],
        'stid': [STID, {'node_dim': 32, 'embed_dim': 1024, 'input_dim': 1, 'num_layer': 1, 'if_node': True}],
        'stnorm': [STNorm, {'tnorm_bool': True, 'snorm_bool': True,
                            'channels': 16, 'kernel_size': 2, 'blocks': 1, 'layers': 2}],
        'magnet': [MAGNet, {'label_window_size': 48, 'conv2d_in_channels': 1, 'residual_channels': 32,
                            'conv_channels': 32, 'skip_channels': 4, 'end_channels': 128, 'node_dim': 40,
                            'tanhalpha': 3.0, 'static_feat': None, 'dilation_exponential': 1,
                            'kernel_size': 7, 'gcn_depth': 2, 'gcn_true': False, 'propalpha': 0.05,
                            'layer_norm_affline': True, 'buildA_true': True, 'predefined_A': None, 'dropout': 0.3}],
        'gwn': [GraphWaveNet, {'out_dim': 1, 'supports': None, 'gcn_bool': True, 'addaptadj': True,
                               'aptinit': None, 'in_dim': 1, 'residual_channels': 32, 'dilation_channels': 32,
                               'skip_channels': 256, 'end_channels': 512, 'kernel_size': 2,
                               'blocks': 1, 'layers': 2, 'dropout': 0.3}],
        'fgnn': [FourierGNN, {'embed_size': 128, 'hidden_size': 256, 'hard_thresholding_fraction': 1,
                              'hidden_size_factor': 1, 'sparsity_threshold': 0.01}],
        'coat': [COAT, {'mode': 'dr', 'activation': 'linear',
                        'use_instance_scale': False, 'dropout_rate': 0.}],
        'gain': [GAIN, {'gat_hidden_size': 64, 'gat_nhead': 128, 'gru_hidden_size': 8, 'gru_num_layers': 1,
                        'cnn_kernel_size': 3, 'cnn_out_channels': 16, 'highway_window_size': 10, 'dropout_rate': 0.5}],
    }

    model_cls, user_settings = modeler['gain']

    common_ds_params = get_common_params(model_cls.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **user_settings}
    model = model_cls(**model_settings)

    print('{}\n{}\n{}'.format(train_ds, val_ds, model))

    model_name = type(model).__name__
    model = covert_parameters(model, torch_float_type)
    print(model_name, count_parameters(model))

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_params, lr=0.001, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.996)

    criterion = nn.MSELoss()
    additive_criterion = getattr(model, 'loss', None)
    evaluator = Evaluator(['MAE', 'RMSE', 'PCC'])

    trainer = Trainer(device, model, is_initial_weights=True,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      criterion=criterion, additive_criterion=additive_criterion, evaluator=evaluator,
                      global_scaler=scaler, global_ex_scaler=ex_scaler)

    trainer.fit(train_ds,
                val_ds,
                epoch_range=(1, 2000), batch_size=32, shuffle=True,
                verbose=True, display_interval=20)

    print('Good luck!')


if __name__ == '__main__':
    initial_seed(2025)
    main()
