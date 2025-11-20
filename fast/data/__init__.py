#!/usr/bin/env python
# encoding: utf-8

"""

    This package contains the data processing modules for time series datasets.

    (1) Scaling techniques.

    (2) Patching techniques.

    (3) Dataset classes for different time series datasets.

"""

from .scale import AbstractScale, MinMaxScale, MeanScale, MaxScale, StandardScale, LogScale
from .scale import InstanceScale, InstanceStandardScale
from .scale import scaler_fit, scaler_transform

from .patch import PatchMaker

from .mask import AbstractMasker, RandomMasker, BlockMasker, VariableMasker, masker_generate

from .sst_dataset import SSTDataset, multi_step_ahead_split, train_test_split
from .smt_dataset import SMTDataset
from .mmt_dataset import MMTDataset
from .smi_dataset import SMIDataset
from .smc_dataset import SMCDataset
from .smir_dataset import SMIrDataset, smir_collate_fn
from .smir_dataset import AbstractSupervisedStrategy, ThresholdSupervisedStrategy, WindowSupervisedStrategy
from .smir_dataset import PairwiseSupervisedStrategy
