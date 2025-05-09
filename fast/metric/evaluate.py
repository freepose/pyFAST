#!/usr/bin/env python
# encoding: utf-8

"""
    The ``Evaluator()`` class manages the metrics for both the loss function and evaluation metrics.
"""

import torch
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod

from .metric import MSE, MAE, RMSE, MAPE, CVRMSE, SMAPE, PCC, RAE


class AbstractEvaluator(ABC):
    """
        Abstract class for streaming aggregated evaluating forecasting models on large-scale datasets.
    """

    def __init__(self):
        self.metrics = {}  # format: {metric_name: metric_class, ...}, default is empty

    @abstractmethod
    def reset(self):
        """ Reset the metrics to its initial state. """
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """ Update the metrics with a new batch of <prediction tensor, real tensor, or mask tensor> pair. """
        pass

    @abstractmethod
    def compute(self) -> Dict:
        """ Compute the evaluation metrics. """
        pass


class EmptyEvaluator(AbstractEvaluator):
    """
        An evaluator that does nothing. Useful for debugging or disabling evaluation.

        Do nothing.
    """
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        pass

    def compute(self) -> Dict:
        return {}


class Evaluator(AbstractEvaluator):
    """

        The ``Evaluator()`` class manages the metrics for both the loss function and evaluation metrics.

        The metrics support both complete and incomplete time series.

        :param metrics: List of **metric names** to use. If None, use all available metrics.
        :param metric_params: Dictionary of metric names and their additional parameters, e.g., PCC bias.
                              {'PCC': {'bias': 0.001}}.
    """

    def __init__(self, metrics: Union[List[str], Tuple[str]] = None, metric_params: dict = None):
        super().__init__()

        self.available_metrics = {
            'MSE': MSE,
            'MAE': MAE,
            'RMSE': RMSE,
            'MAPE': MAPE,
            'sMAPE': SMAPE,
            'CV-RMSE': CVRMSE,
            'PCC': PCC,
            'RAE': RAE,
            # TODO: Add more metrics here as needed
        }

        if metrics is None:
            self.metrics = self.available_metrics
        else:
            # Ensure all specified metrics are valid
            assert all(metric in self.available_metrics for metric in metrics), \
                f'Metrics should be in {list(self.available_metrics.keys())}'

            # Filter the available metrics based on the provided list
            self.metrics = {metric: self.available_metrics[metric] for metric in metrics if
                            metric in self.available_metrics}

        self.metric_params = metric_params if metric_params else {}

        self.metric_instances = {name: metric_class(**self.metric_params.get(name, {}))
                                 for name, metric_class in self.metrics.items()}

    def reset(self):
        """ Reset all metrics to their initial state. """
        for name, metric_inst in self.metric_instances.items():
            metric_inst.reset()

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metrics with a new batch of <prediction tensor, real tensor, or mask tensor> pair.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        for name, metric_inst in self.metric_instances.items():
            metric_inst.update(prediction, real, mask)

    def compute(self) -> Dict:
        """
            Evaluate the prediction performance of error metrics.
            :return: Dictionary of metric names and their calculated values.
        """
        results = {}
        for name, metric_inst in self.metric_instances.items():
            ret = metric_inst.compute()
            results[name] = float(ret)
        return results

    def evaluate(self, *tensors: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Dict:
        """
            Evaluate the prediction performance using the error / accuracy metrics.

            :param tensors: prediction tensor, real value tensor, (maybe) mask tensor (1d, 2d, or 3d tensor)
            :return: dictionary of metric names and their calculated values.
        """
        results = {}
        for name, metric_inst in self.metric_instances.items():
            params = self.metric_params.get(name, {})
            ret = metric_inst(*tensors, **params)
            results[name] = float(ret)
        return results
