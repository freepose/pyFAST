#!/usr/bin/env python
# encoding: utf-8

# Dense target variable + dense exogenous variables

from .ex.narx import ARX, NARXMLP, NARXRNN
from .ex.dsar import DSAR
from .ex.dgr import DGR
from .ex.mvt import MvT
from .ex.dgdr import DGDR
from .ex.gainge import GAINGE
from .ex.cabin import Cabin
from .ex.tspt import TSPT
from .ex.temporalcausalnet import TemporalCausalNet

# Dense target variable + preknown exogenous variables
from .ex2.transformer_ex2 import TransformerEx2
from .ex2.informer_ex2 import InformerEx2
from .ex2.autoformer_ex2 import AutoformerEx2
from .ex2.fedformer_ex2 import FEDformerEx2
from .ex2.deep_time import DeepTIMe

# Sparse target variable + dense exogenous variables + pre-known exogenous variables (e.g., time)

# Plugins
from .fusion_plugin import DataFirstPlugin, LearningFirstPlugin, ExogenousDataDrivenPlugin
