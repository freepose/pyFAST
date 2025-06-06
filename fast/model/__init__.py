#!/usr/bin/env python
# encoding: utf-8

"""

    All modules / models are imported here.

    Model names are used as keys to access the model classes.

"""

from .mts import GAR, AR, VAR, ANN, TimeSeriesRNN, EncoderDecoder
from .mts import NHiTS, DLinear, NLinear
from .mts import TemporalConvNet, CNN1D, CNNRNN, CNNRNNRes, LSTNet
from .mts import DeepResidualNetwork
from .mts import Transformer, Informer, Autoformer, FiLM, Triformer, FEDformer, Crossformer
from .mts import TimesNet, PatchTST, STAEformer, iTransformer, TimeXer, TimeMixer
from .mts import TimesFM, Timer
from .mts import STID, STNorm, MAGNet, GraphWaveNet, FourierGNN, AGCRN
from .mts import COAT, TCOAT, CoDR, CTRL, GAIN, DRED, AFTS

from .mts_fusion import ARX, NARXMLP, NARXRNN
from .mts_fusion import DSAR, DGR, MvT, DGDR, GAINGE, TSPT, TemporalCausalNet
from .mts_fusion import DataFirstPlugin, LearningFirstPlugin, ExogenousDataDrivenPlugin

"""
    Time series forecasting models. They support both univariate and multivariate time series forcasting.
"""

ts_model_classes = {
    'gar': GAR,
    'ar': AR,
    'var': VAR,
    'ann': ANN,
    'drn': DeepResidualNetwork,
    'rnn': TimeSeriesRNN,  # it includes rnn, lstm, gru, minlstm
    'ed': EncoderDecoder,  # it includes rnn, lstm, gru, minlstm
    'tcn': TemporalConvNet,
    'cnn1d': CNN1D,
    'cnnrnn': CNNRNN,
    'cnnrnnres': CNNRNNRes,
    'lstnet': LSTNet,
    'nhits': NHiTS,
    'nlinear': NLinear,
    'dlinear': DLinear,

    # Transformer models
    'transformer': Transformer,
    'informer': Informer,
    'autoformer': Autoformer,
    'fedformer': FEDformer,
    'film': FiLM,
    'triformer': Triformer,
    'crossformer': Crossformer,
    'timesnet': TimesNet,
    'patchtst': PatchTST,
    'staeformer': STAEformer,
    'itransformer': iTransformer,
    'timesfm': TimesFM,
    'timer': Timer,
    'timexer': TimeXer,
    'timemixer': TimeMixer,

    # Graph neural networks
    'stid': STID,
    'stnorm': STNorm,
    'magnet': MAGNet,
    'gwn': GraphWaveNet,
    'fgnn': FourierGNN,
    'agcrn': AGCRN,

    # Our models
    'gain': GAIN,
    'coat': COAT,
    'tcoat': TCOAT,
    'codr': CoDR,
    'ctrl': CTRL,
    'dred': DRED,
    'uninet': AFTS,

}

"""
    Time series forecasting using target variables and exogenous variables.
    Both target variable and exogenous variables are dense data (i.e., not sparse time series).
"""

ts_ex_model_classes = {
    'dsar': DSAR,
    'dgr': DGR,
    'mvt': MvT,
    'dgdr': DGDR,
    'gainge': GAINGE,
    'tspt': TSPT,
    'tcausn': TemporalCausalNet,
}

ts_ex_plugin_classes = {
    'dfp': DataFirstPlugin,
    'lfp': LearningFirstPlugin,
    'exdd': ExogenousDataDrivenPlugin,  # forecasting solely based on exogenous variables
}

ts_model_classes.update(ts_ex_model_classes)
ts_model_classes.update(ts_ex_plugin_classes)
