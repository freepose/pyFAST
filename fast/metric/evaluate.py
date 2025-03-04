#!/usr/bin/env python
# encoding: utf-8

"""
    The Evaluator supports a set of metric functions.
"""
import torch

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
from .mask_metric import mask_pearson_correlation_coefficient


class Evaluator:
    def __init__(self, metrics: list[str] or tuple[str] = None, metric_params: dict = None):
        """
            Initialize the Evaluator with a list of metrics and their parameters.
            The metrics support both complete and incomplete time series.

            :param metrics: List of metric names to use. If None, use all available metrics.
            :param metric_params: Dictionary of metric names and their additional parameters.
        """
        self.available_metrics = {
            'MAE': mean_absolute_error,
            'MSE': mean_squared_error,
            'RMSE': root_mean_squared_error,
            'CV-RMSE': coefficient_of_variation_of_RMSE,
            'RMSPE': root_mean_square_percentage_error,
            'RMdSPE': root_median_square_percentage_error,
            'MAPE': mean_absolute_percentage_error,
            'sMAPE': symmetric_mean_absolute_percentage_error,
            'MdAPE': median_absolute_percentage_error,
            'sMdAPE': symmetric_median_absolute_percentage_error,
            'SDRE': standard_deviation_relative_errors,
            'cutMAPE': cutoff_mean_absolute_percentage_error,
            'RAE': relative_absolute_error,
            'RSE': relative_squared_error,
            'PCC': empirical_correlation_coefficient,
            'R2': r_square,
            'MASE': mean_absolute_scaled_error,
            'MdASE': median_absolute_scaled_error,
            'maskMAE': mask_mean_absolute_error,
            'maskMSE': mask_mean_squared_error,
            'maskRMSE': mask_root_mean_squared_error,
            'maskMAPE': mask_mean_absolute_percentage_error,
            'masksMAPE': mask_symmetric_mean_absolute_percentage_error,
            'maskPCC': mask_pearson_correlation_coefficient
        }

        if metrics is None:
            self.metric_dict = self.available_metrics
        else:
            assert all(metric in self.available_metrics for metric in metrics), \
                f'Metrics should be in {list(self.available_metrics.keys())}'

            self.metric_dict = {metric: self.available_metrics[metric] for metric in metrics if
                                metric in self.available_metrics}

        self.metric_params = metric_params if metric_params is not None else {}

    def evaluate(self, *tensors: tuple[torch.Tensor] or list[torch.Tensor]) -> dict:
        """
            Evaluate the prediction performance using the error / accuracy metrics.

            :param tensors: prediction values, real values, (maybe) mask tensor (1d, 2d, or 3d numpy array)
            :return: dictionary of metric names and their calculated values.
        """
        results = {}
        for name, func in self.metric_dict.items():
            params = self.metric_params.get(name, {})
            ret = func(*tensors, **params)
            results[name] = float(ret)
        return results
