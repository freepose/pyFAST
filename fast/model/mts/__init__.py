#!/usr/bin/env python
# encoding: utf-8

from .ar import GAR, AR, VAR, ANN
from .rnn import TimeSeriesRNN, EncoderDecoder
from .tcn import TemporalConvNet
from .cnn import CNNRNN, CNNRNNRes
from .lstnet import LSTNet
from .nhits import NHiTS
from .dlinear import DLinear, NLinear
from .drn import DeepResidualNetwork
from .transformer.transformer import Transformer
from .transformer.informer import Informer
from .transformer.autoformer import Autoformer
from .transformer.fedformer import FEDformer
from .transformer.film import FiLM
from .transformer.triformer import Triformer
from .transformer.crossformer import Crossformer
from .transformer.timesnet import TimesNet
from .transformer.patchtst import PatchTST
from .transformer.staeformer import STAEformer
from .transformer.itransformer import iTransformer
from .transformer.timesfm import TimesFM
from .transformer.timer import Timer
from .transformer.timexer import TimeXer
from .transformer.timemixer import TimeMixer

from .gnn.stid import STID
from .gnn.stnorm import STNorm
from .gnn.magnat import MAGNet
from .gnn.gwn import GraphWaveNet
from .gnn.fgnn import FourierGNN
from .gnn.gain import GAIN

from .coat import COAT, TCOAT, CoDR, CTRL
from .dred import DRED
