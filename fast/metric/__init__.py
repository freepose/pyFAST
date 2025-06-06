#!/usr/bin/env python
# encoding: utf-8

"""

    The ``metric`` package provides error metrics, which are used as loss function, or error metrics.

    (1) The metrics work with streaming aggregation of performances of batches.

    (2) The metrics work with mask real values to ignore missing values.

"""

# Metrics
from .metric import AbstractMetric, MSE, MAE, RMSE, MAPE, SMAPE, CVRMSE, RAE, RSE, R2, PCC

# Evaluator
from .evaluate import AbstractEvaluator, EmptyEvaluator, Evaluator
