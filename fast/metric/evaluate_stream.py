#!/usr/bin/env python
# encoding: utf-8

import torch
from typing import List, Tuple, Dict

from .stream_metric import StreamMSE, StreamMAE, StreamRMSE, StreamMAPE, StreamCVRMSE, StreamSMAPE


class StreamEvaluator:
    """
        The metrics support both complete and incomplete time series.

        :param metrics: List of metric names to use. If None, use all available metrics.
        :param metric_params: Dictionary of metric names and their additional parameters.
    """

    def __init__(self, metrics: List[str] or Tuple[str] = None, metric_params: dict = None):
        self.available_metrics = {
            'MSE': StreamMSE,
            'MAE': StreamMAE,
            'RMSE': StreamRMSE,
            'MAPE': StreamMAPE,
            'sMAPE': StreamSMAPE,
            'CV-RMSE': StreamCVRMSE,
            # Add more metrics here as needed
        }

        if metrics is None:
            self.metric_dict = self.available_metrics
        else:
            # Ensure all specified metrics are valid
            assert all(metric in self.available_metrics for metric in metrics), \
                f'Metrics should be in {list(self.available_metrics.keys())}'

            # Filter the available metrics based on the provided list
            self.metric_dict = {metric: self.available_metrics[metric] for metric in metrics if
                                metric in self.available_metrics}

        # Store additional parameters for each metric
        self.metric_params = metric_params if metric_params is not None else {}

        # Instantiate the metrics
        for name, metric_class in self.metric_dict.items():
            # Initialize the metric class with any additional parameters
            if name in self.metric_params:
                metric_inst = metric_class(**self.metric_params[name])
            else:
                metric_inst = metric_class()

            # Store the instantiated metric
            self.metric_dict[name] = metric_inst

    def reset(self):
        """ Reset all metrics to their initial state. """
        for name, metric_inst in self.metric_dict.items():
            metric_inst.reset()

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        for name, metric_inst in self.metric_dict.items():
            metric_inst.update(prediction, real, mask)

    def compute(self) -> dict:
        """
            Evaluate the prediction performance of error metrics.
            :return: Dictionary of metric names and their calculated values.
        """
        results = {}
        for name, metric_inst in self.metric_dict.items():
            ret = metric_inst.compute()
            results[name] = float(ret)
        return results