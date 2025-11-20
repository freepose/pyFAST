#!/usr/bin/env python
# encoding: utf-8

"""
    The package supports sparse_fusion situations for target variables and exogenous variables.

    (1) The target variables are ex_mask.

    (2)
"""

from .ex2.tpatchgnn import TPatchGNN
from .ex2.cru import CRU

from .ex2.transformer_mask_ex2 import TransformerMaskEx2
from .ex2.informer_mask_ex2 import InformerMaskEx2
from .ex2.autoformer_mask_ex2 import AutoformerMaskEx2
from .ex2.fedformer_mask_ex2 import FEDformerMaskEx2
