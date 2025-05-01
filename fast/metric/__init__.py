#!/usr/bin/env python
# encoding: utf-8

"""
    This package provides both error metrics and mask error metrics calculations.
"""

from .metric import mean_squared_error, root_mean_squared_error, coefficient_of_variation_of_RMSE
from .metric import standard_deviation_relative_errors, root_mean_square_percentage_error
from .metric import mean_absolute_error, mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error
from .metric import median_absolute_percentage_error, root_median_square_percentage_error
from .metric import symmetric_median_absolute_percentage_error
from .metric import r_square, relative_absolute_error, relative_squared_error, empirical_correlation_coefficient
from .metric import cutoff_mean_absolute_percentage_error
from .metric import mean_absolute_scaled_error, median_absolute_scaled_error

from .mask_metric import mask_mean_absolute_error, mask_mean_squared_error, mask_root_mean_squared_error
from .mask_metric import mask_mean_absolute_percentage_error, mask_symmetric_mean_absolute_percentage_error

from .stream_metric import StreamMSE, StreamMAE

from .evaluate import Evaluator
from .evaluate_stream import StreamEvaluator
