#!/usr/bin/env python
# encoding: utf-8

import torch
from abc import abstractmethod, ABC


class AbstractStreamMetric(object):
    """ Do nothing. """

    def __init__(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *tensors):
        pass

    @abstractmethod
    def compute(self):
        pass


class StreamMSE(AbstractStreamMetric):
    """
        Mean Squared Error (MSE).
    """

    def __init__(self):
        super().__init__()
        self.sum_squared_errors = 0.0
        self.total_samples = 0

    def reset(self):
        self.sum_squared_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2

        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.total_samples += real.numel()

    def compute(self):
        """
            :return: Mean Squared Error (MSE)
        """
        if self.total_samples == 0:
            return 0.0
        return self.sum_squared_errors / self.total_samples


class StreamMAE(AbstractStreamMetric):
    """
        Mean Absolute Error (MAE).
    """

    def __init__(self):
        super().__init__()
        self.sum_absolute_errors = 0.0
        self.total_samples = 0

    def reset(self):
        self.sum_absolute_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        absolute_errors = torch.abs(prediction - real)

        self.sum_absolute_errors += absolute_errors.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self):
        """
            :return: Mean Absolute Error (MAE)
        """
        if self.total_samples == 0:
            return 0.0
        return self.sum_absolute_errors / self.total_samples


class StreamRMSE(AbstractStreamMetric):
    """
        Root Mean Squared Error (RMSE).
    """

    def __init__(self):
        super().__init__()
        self.sum_squared_errors = 0.0
        self.total_samples = 0

    def reset(self):
        self.sum_squared_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2

        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.total_samples += real.numel()

    def compute(self):
        """
            :return: Root Mean Squared Error (RMSE)
        """
        if self.total_samples == 0:
            return 0.0
        _rmse = (self.sum_squared_errors / self.total_samples) ** 0.5

        return _rmse


class StreamMAPE(AbstractStreamMetric):
    """
        Mean Absolute Percentage Error (MAPE).
    """

    def __init__(self):
        super().__init__()
        self.sum_absolute_percentage_errors = 0.0
        self.total_samples = 0

    def reset(self):
        self.sum_absolute_percentage_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        absolute_percentage_errors = torch.abs((prediction - real) / (real + 1e-8))

        self.sum_absolute_percentage_errors += absolute_percentage_errors.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self):
        """
            :return: Mean Absolute Percentage Error (MAPE)
        """

        if self.total_samples == 0:
            return 0.0
        return self.sum_absolute_percentage_errors / self.total_samples


class StreamSMAPE(AbstractStreamMetric):
    """
        Symmetric Mean Absolute Percentage Error (sMAPE).
    """

    def __init__(self):
        super().__init__()
        self.sum_ape = 0.0
        self.total_samples = 0

    def reset(self):
        self.sum_ape = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        errors = real - prediction
        ape = (errors / (real + prediction)).abs()

        self.sum_ape += ape.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self):
        """
            :return: Symmetric Mean Absolute Percentage Error (SMAPE)
        """
        if self.total_samples == 0:
            return 0.0
        smape = (self.sum_ape / self.total_samples) * 2

        return smape


class StreamCVRMSE(AbstractStreamMetric):
    """
        Streaming aggregated coefficient of variation of RMSE (CV-RMSE). It works on weighted sum.
    """

    def __init__(self):
        super().__init__()
        self.sum_squared_errors = 0.0
        self.sum_real_values = 0.0
        self.total_samples = 0

    def reset(self):
        self.sum_squared_errors = 0.0
        self.sum_real_values = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.
            :param prediction: predictions tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param real: target tensor, shape is ``(batch_size, seq_len, n_vars)``
            :param mask: mask tensor, shape is ``(batch_size, seq_len, n_vars)``.
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2

        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.sum_real_values += real.sum().detach().item()
        self.total_samples += real.numel()

    def compute(self):
        """
            :return: Coefficient of Variation of RMSE (CV-RMSE)
        """
        if self.total_samples == 0:
            return 0.0
        mean_real = self.sum_real_values / self.total_samples
        rmse = (self.sum_squared_errors / self.total_samples) ** 0.5

        return rmse / mean_real
