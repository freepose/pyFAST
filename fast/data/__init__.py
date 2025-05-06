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
from .scale import scale_several_time_series

from .patch import PatchMaker

from .sst_dataset import SSTDataset, multi_step_ahead_split, train_test_split
from .smt_dataset import SMTDataset
from .mmt_dataset import MMTDataset
from .bdp_dataset import BDPDataset, bdp_collate_fn
