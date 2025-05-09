#!/usr/bin/env python
# encoding: utf-8

"""
    Univariate Time Series Modeling
"""

from fast.model.mts import GAR, AR, VAR, ANN, TimeSeriesRNN, EncoderDecoder
from fast.model.mts import NHiTS, DLinear, NLinear
from fast.model.mts import TemporalConvNet, CNN1D, CNNRNN, CNNRNNRes, LSTNet
from fast.model.mts import DeepResidualNetwork
from fast.model.mts import Transformer, Informer, Autoformer, FiLM, Triformer, FEDformer, Crossformer
from fast.model.mts import TimesNet, PatchTST, STAEformer, iTransformer, TimeXer, TimeMixer
from fast.model.mts import TimesFM, Timer, LSTD
from fast.model.mts import STID, STNorm, MAGNet, GraphWaveNet, FourierGNN
from fast.model.mts import GAIN, AGCRN, COAT, TCOAT, CoDR

etth1_univariate_modeler = {
    'gar': [GAR, {'activation': 'linear'}],
    'ar': [AR, {'activation': 'linear'}],
    'var': [VAR, {'activation': 'linear'}],
    'ann': [ANN, {'hidden_size': 512}],
    'drn': [DeepResidualNetwork, {'hidden_size': 64, 'number_stacks': 7, 'number_blocks_per_stack': 1,
                                  'use_rnn': False}],
    'rnn': [TimeSeriesRNN, {'rnn_cls': 'rnn', 'hidden_size': 16, 'num_layers': 1,
                            'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
    'lstm': [TimeSeriesRNN, {'rnn_cls': 'lstm', 'hidden_size': 16, 'num_layers': 1,
                             'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
    'gru': [TimeSeriesRNN, {'rnn_cls': 'gru', 'hidden_size': 16, 'num_layers': 1,
                            'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
    'ed': [EncoderDecoder, {'rnn_cls': 'gru', 'hidden_size': 32, 'num_layers': 1,
                            'bidirectional': False, 'dropout_rate': 0., 'decoder_way': 'mapping'}],
    'cnn1d': [CNN1D, {'out_channels': 1, 'kernel_size': 1}],  # It's a linear model.
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
    'nlinear': [NLinear, {'individual': False}],
    'dlinear': [DLinear, {'individual': False, 'kernel_size': 5}],
    'transformer': [Transformer, {'label_window_size': 48, 'd_model': 512, 'num_heads': 8,
                                  'num_encoder_layers': 2, 'num_decoder_layers': 1,
                                  'dim_ff': 2048, 'dropout_rate': 0.05}],
    'informer': [Informer, {'label_window_size': 48, 'd_model': 512, 'num_heads': 8,
                            'num_encoder_layers': 2, 'num_decoder_layers': 1,
                            'dim_ff': 2048, 'dropout_rate': 0.05}],
    'autoformer': [Autoformer, {'label_window_size': 48, 'd_model': 512, 'num_heads': 8,
                                'num_encoder_layers': 2, 'num_decoder_layers': 1,
                                'dim_ff': 2048, 'dropout_rate': 0., 'moving_avg': 7}],
    'fedformer': [FEDformer, {'label_window_size': 48, 'd_model': 512, 'num_heads': 8,
                              'num_encoder_layers': 2, 'num_decoder_layers': 1,
                              'dim_ff': 2048, 'activation': 'relu', 'moving_avg': 7, 'dropout_rate': 0.05,
                              'version': 'fourier', 'mode_select': 'random', 'modes': 32}],
    'film': [FiLM, {'d_model': 512, 'use_instance_scale': True}],  # 'MPS' not works
    'triformer': [Triformer, {'channels': 32, 'patch_sizes': [8, 3], 'mem_dim': 5}],
    'crossformer': [Crossformer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 1, 'num_decoder_layers': 1,
                                  'dim_ff': 2048, 'dropout_rate': 0.05, 'factor': 5,
                                  'seg_len': 24, 'win_size': 2}],
    'timesnet': [TimesNet, {'d_model': 512, 'num_encoder_layers': 2, 'dim_ff': 2048, 'dropout_rate': 0.05,
                            'num_kernels': 6, 'top_k': 5, 'use_instance_scale': False}],
    'patchtst': [PatchTST, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 6, 'dim_ff': 2048,
                            'patch_len': 24, 'patch_stride': 12, 'patch_padding': 0,
                            'use_instance_scale': True}],
    'staeformer': [STAEformer, {'input_dim': 1, 'output_dim': 1, 'input_embedding_dim': 24,
                                'tod_steps_per_day': 24, 'tod_embedding_dim': 0, 'dow_embedding_dim': 0,
                                'spatial_embedding_dim': 0, 'adaptive_embedding_dim': 80,
                                'dim_ff': 2048, 'num_heads': 8, 'num_layers': 2,
                                'dropout_rate': 0.05, 'use_mixed_proj': True}],
    'itransformer': [iTransformer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 2, 'dim_ff': 2048,
                                    'activation': 'relu', 'dropout_rate': 0.05, 'use_instance_scale': True}],
    'timesfm': [TimesFM, {'d_model': 512, 'num_heads': 2, 'num_layers': 2,
                          'dim_ff': 2048, 'dropout_rate': 0.05}],
    'timer': [Timer, {'patch_len': 4, 'd_model': 512, 'num_heads': 8, 'e_layers': 1, 'dim_ff': 2048,
                      'activation': 'relu', 'dropout_rate': 0.05}],
    'timexer': [TimeXer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 2,
                          'dim_ff': 2048, 'dropout_rate': 0.05, 'activation': 'relu',
                          'patch_len': 16, 'use_instance_scale': True}],
    'timemixer': [TimeMixer, {'d_model': 512, 'num_heads': 8, 'num_encoder_layers': 1,
                              'dim_ff': 2048, 'dropout_rate': 0.05, 'moving_avg': 25,
                              'top_k': 5, 'channel_independence': True, 'decomposition_method': 'moving_avg',
                              'down_sampling_method': 'avg', 'down_sampling_window': 1, 'down_sampling_layers': 1,
                              'use_instance_scale': True}],
    'lstd': [LSTD, {'label_window_size': 24, 'latent_dim_d': 32, 'latent_dim_s': 32, 'hidden_dim': 32}],
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
    'gain': [GAIN, {'gat_hidden_size': 64, 'gat_nhead': 128, 'gru_hidden_size': 8, 'gru_num_layers': 1,
                    'cnn_kernel_size': 3, 'cnn_out_channels': 16, 'highway_window_size': 10, 'dropout_rate': 0.05}],
    'agcrn': [AGCRN, {'input_dim': 1, 'output_dim': 1, 'rnn_units': 32, 'num_layers': 2, 'default_graph': True,
                      'embed_dim': 3, 'cheb_k': 2}],
    'coat': [COAT, {'mode': 'dr', 'activation': 'linear', 'use_instance_scale': False, 'dropout_rate': 0.05}],
    'tcoat': [TCOAT, {'rnn_hidden_size': 16, 'rnn_num_layers': 1, 'rnn_bidirectional': False,
                      'residual_window_size': 24, 'residual_ratio': 1.0, 'dropout_rate': 0.0}],
    'codr': [CoDR, {'horizon': 1, 'hidden_size': 64, 'use_window_fluctuation_extraction': True,
                    'dropout_rate': 0.2}],
}

etth2_univariate_modeler = etth1_univariate_modeler

ettm1_univariate_modeler = {

}

ettm2_univariate_modeler = ettm1_univariate_modeler

"""
    Multivariate Time Series Modeling
"""

etth1_multivariate_modeler = {

}

etth2_multivariate_modeler = etth1_multivariate_modeler

ettm1_multivariate_modeler = {

}

ettm2_multivariate_modeler = ettm1_multivariate_modeler
