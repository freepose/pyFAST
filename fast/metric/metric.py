#!/usr/bin/env python
# encoding: utf-8

"""

    Streaming aggregated metrics for large-scale dataset evaluation.
    The global metric values are calculated by aggregating batchify mediate updates.

    This also provides global evaluation with streaming aggregation.

    Note: The assert statement checks are removed to avoid unnecessary overhead in the call method.

"""

import torch
from abc import abstractmethod, ABC


class AbstractMetric(ABC):
    """
        Abstract class for streaming aggregated metrics.
    """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def compute(self):
        pass


class MSE(AbstractMetric):
    """
        Mean Squared Error (MSE). This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_squared_errors: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            MSE. Element-wise metrics.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
            :return             MSE value.
        """

        # The assert statement checks are removed to avoid unnecessary overhead in the call method.
        # assert prediction.shape == real.shape, 'preds tensor and real tensor must have the same shape'

        if mask is not None:
            # The assert statement checks are removed to avoid unnecessary overhead in the call method.
            # assert real.shape == mask.shape, 'real tensor and mask tensor must have the same shape'

            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2
        mse = squared_errors.mean()

        return mse

    def reset(self):
        self.sum_squared_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2

        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.total_samples += real.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Mean Squared Error (MSE)
        """
        if self.total_samples == 0:
            return 0.0
        return self.sum_squared_errors / self.total_samples


class MAE(AbstractMetric):
    """
        Mean Absolute Error (MAE). This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_absolute_errors: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            MAE. Element-wise metrics.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        absolute_errors = (prediction - real).abs()
        mae = absolute_errors.mean()

        return mae

    def reset(self):
        self.sum_absolute_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        absolute_errors = torch.abs(prediction - real)

        self.sum_absolute_errors += absolute_errors.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Mean Absolute Error (MAE)
        """
        if self.total_samples == 0:
            return 0.0
        return self.sum_absolute_errors / self.total_samples


class MRE(AbstractMetric):
    """
        Mean Relative Error (MRE). This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).

        Division by zero is not explicitly handled; users should ensure real values are non-zero.
        This metric is not fitted for datasets with zero or near-zero real values.

    """

    def __init__(self):
        self.sum_relative_errors: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            MRE. Element-wise metrics.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
            :return:            MRE value.
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        relative_errors = ((prediction - real) / real).abs()
        mre = relative_errors.mean()

        return mre

    def reset(self):
        self.sum_relative_errors = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        relative_errors = ((prediction - real) / real).abs()

        self.sum_relative_errors += relative_errors.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Mean Relative Error (MRE)
        """
        if self.total_samples == 0:
            return 0.0
        return self.sum_relative_errors / self.total_samples


class RMSE(AbstractMetric):
    """
        Root Mean Squared Error (RMSE). This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_squared_errors: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            RMSE. Element-wise metrics.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
            :return: RMSE value.
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2
        rmse = squared_errors.mean().sqrt()

        return rmse

    def reset(self):
        self.sum_squared_errors: float = 0.0
        self.total_samples: int = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2

        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.total_samples += real.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Root Mean Squared Error (RMSE)
        """
        if self.total_samples == 0:
            return 0.0
        rmse = (self.sum_squared_errors / self.total_samples) ** 0.5

        return rmse


class MAPE(AbstractMetric):
    """
        Mean Absolute Percentage Error (MAPE). This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_absolute_percentage_errors: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            MAPE. Element-wise metrics.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
            :return:            MAPE value.
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        absolute_percentage_errors = ((prediction - real) / real).abs()
        mape = absolute_percentage_errors.mean()

        return mape

    def reset(self):
        self.sum_absolute_percentage_errors: float = 0.0
        self.total_samples: int = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        absolute_percentage_errors = ((prediction - real) / real).abs()

        self.sum_absolute_percentage_errors += absolute_percentage_errors.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Mean Absolute Percentage Error (MAPE)
        """

        if self.total_samples == 0:
            return 0.0
        return self.sum_absolute_percentage_errors / self.total_samples


class SMAPE(AbstractMetric):
    """
        Symmetric Mean Absolute Percentage Error (sMAPE). This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_ape: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            sMAPE. Element-wise metrics.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        symmetric_ape = ((real - prediction) / (real + prediction)).abs()
        smape = symmetric_ape.mean()

        return smape

    def reset(self):
        self.sum_ape = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        symmetric_ape = ((real - prediction) / (real + prediction)).abs()

        self.sum_ape += symmetric_ape.sum().detach().item()
        self.total_samples += prediction.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Symmetric Mean Absolute Percentage Error (SMAPE)
        """
        if self.total_samples == 0:
            return 0.0
        smape = (self.sum_ape / self.total_samples) * 2

        return smape


class CVRMSE(AbstractMetric):
    """
        Streaming aggregated coefficient of variation of RMSE (CV-RMSE).
        This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_squared_errors: float = 0.0
        self.sum_real_values: float = 0.0
        self.total_samples: int = 0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            CV-RMSE value.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        rmse = ((prediction - real) ** 2).mean().sqrt()
        cvrmse = rmse / real.mean()

        return cvrmse

    def reset(self):
        self.sum_squared_errors = 0.0
        self.sum_real_values = 0.0
        self.total_samples = 0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Update the metric with a batch of predictions and targets.

            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
        """
        assert prediction.shape == real.shape, 'preds and real must have the same shape'

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2

        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.sum_real_values += real.sum().detach().item()
        self.total_samples += real.numel()

    def compute(self) -> float:
        """
            :return: Aggregated Coefficient of Variation of RMSE (CV-RMSE)
        """
        if self.total_samples == 0:
            return 0.0
        mean_real = self.sum_real_values / self.total_samples
        rmse = (self.sum_squared_errors / self.total_samples) ** 0.5
        cvrmse = rmse / mean_real

        return cvrmse


class RAE(AbstractMetric):
    """
        Relative Absolute Error (RAE). Computes the sum of absolute errors
        relative to the sum of absolute deviations from the mean of the actual values.

        Streaming aggregated RAE.
        This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_abs_errors: float = 0.0
        self.sum_abs_deviation: float = 0.0
        self.targets = []

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            RAE. Element-wise metrics.
            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
            :return: RAE value.
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        abs_errors = (real - prediction).abs()
        abs_deviation = (real - real.mean()).abs()
        rae = abs_errors.sum() / abs_deviation.sum()
        return rae

    def reset(self):
        self.sum_abs_errors = 0.0
        self.sum_abs_deviation = 0.0
        self.targets = []

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        abs_errors = (prediction - real).abs()
        self.sum_abs_errors += abs_errors.sum().detach().item()

        self.targets.append(real.detach())

    def compute(self) -> float:
        if not self.targets:
            return 0.0

        all_targets = torch.cat(self.targets)
        mean_real = all_targets.mean()
        abs_deviation = torch.abs(all_targets - mean_real)
        self.sum_abs_deviation = abs_deviation.sum().item()

        if self.sum_abs_deviation == 0:
            return 0.0  # or float('inf') depending on how you want to handle perfect constancy

        return self.sum_abs_errors / self.sum_abs_deviation


class RSE(AbstractMetric):
    """
        Relative Squared Error (RSE). Computes the sum of squared errors
        relative to the sum of squared deviations from the mean of the actual values.
        Streaming aggregated RSE.

        This class supports both element-wise evaluation,
        and batch-wise aggregated evaluation on large-scale dataset (prediction values and real values).
    """

    def __init__(self):
        self.sum_squared_errors: float = 0.0
        self.sum_squared_deviation: float = 0.0
        self.targets = []
        self.mean: float = 0.0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            RSE. Element-wise metrics.
            :param prediction:  predicted values (1d, 2d, or 3d torch tensor).
            :param real:        real values (1d, 2d, or 3d torch tensor).
            :param mask:        mask tensor (1d, 2d, or 3d torch tensor).
            :return: RSE value.
        """

        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (real - prediction) ** 2
        squared_deviation = (real - real.mean()) ** 2
        rse = squared_errors.sum() / squared_deviation.sum()
        return rse

    def reset(self):
        self.sum_squared_errors = 0.0
        self.sum_squared_deviation = 0.0
        self.targets = []
        self.mean: float = 0.0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2
        self.sum_squared_errors += squared_errors.sum().detach().item()

        self.targets.append(real.detach())

    def compute(self) -> float:
        if not self.targets:
            return 0.0

        all_targets = torch.cat(self.targets)
        mean_real = all_targets.mean()
        squared_deviation = (all_targets - mean_real) ** 2
        self.sum_squared_deviation = squared_deviation.sum().item()

        if self.sum_squared_deviation == 0:
            return 0.0  # or float('inf') depending on how you want to handle perfect constancy

        return self.sum_squared_errors / self.sum_squared_deviation


class R2(AbstractMetric):
    """
    R-squared (coefficient of determination).
    Measures the proportion of variance in the target variable that is predictable from the independent variables.

    Supports element-wise calculation and streaming/batch-wise aggregation.
    """

    def __init__(self):
        self.sum_squared_errors: float = 0.0
        self.sum_squared_deviation: float = 0.0
        self.targets = []

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        R^2. Element-wise metric.
        :param prediction: predicted values.
        :param real:       actual values.
        :param mask:       optional mask tensor.
        :return: R^2 score (scalar tensor).
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (real - prediction) ** 2
        squared_deviation = (real - real.mean()) ** 2

        if squared_deviation.sum() == 0:
            return torch.tensor(0.0)  # or float('nan') for undefined

        r2 = 1 - squared_errors.sum() / squared_deviation.sum()
        return r2

    def reset(self):
        self.sum_squared_errors = 0.0
        self.sum_squared_deviation = 0.0
        self.targets = []

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        squared_errors = (prediction - real) ** 2
        self.sum_squared_errors += squared_errors.sum().detach().item()
        self.targets.append(real.detach())

    def compute(self) -> float:
        if not self.targets:
            return 0.0

        all_targets = torch.cat(self.targets)
        mean_real = all_targets.mean()
        squared_deviation = (all_targets - mean_real) ** 2
        self.sum_squared_deviation = squared_deviation.sum().item()

        if self.sum_squared_deviation == 0:
            return 0.0  # or float('nan') if variance is zero

        return 1 - self.sum_squared_errors / self.sum_squared_deviation


class SDRE(AbstractMetric):
    """
        Standard Deviation of Relative Errors (SDRE).
        Calculates standard deviation in a streaming fashion (no storage of intermediate values).
    """

    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.M2: float = 0.0  # Sum of squares of differences from the current mean

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            Element-wise SDRE (not aggregated).
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        epsilon = 1e-8
        relative_errors = (prediction - real) / (real + epsilon)
        return torch.std(relative_errors)

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
            Streaming update using Welford’s algorithm for standard deviation.
        """
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        epsilon = 1e-8
        relative_errors = ((prediction - real) / (real + epsilon)).detach().flatten()

        for x in relative_errors:
            self.n += 1
            delta = x.item() - self.mean
            self.mean += delta / self.n
            delta2 = x.item() - self.mean
            self.M2 += delta * delta2

    def compute(self) -> float:
        """
            Final computation of streaming SDRE.
        """
        if self.n < 2:
            return 0.0
        variance = self.M2 / (self.n - 1)
        return variance ** 0.5


class PCC(AbstractMetric):
    """
    Streaming Pearson Correlation Coefficient (PCC).
    Supports minibatch updates and full-batch one-shot computation.

    Maintains running sums to compute covariance and variances without
    storing all samples.
    """

    def __init__(self):
        self.total_samples = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def __call__(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            PCC. Element-wise metrics.
            :param prediction:  Predicted values (1d, 2d, or 3d torch tensor)
            :param real:        Real values (1d, 2d, or 3d torch tensor)
            :param mask:        Mask indicator of real values (1d, 2d, or 3d torch tensor)
            :return:           PCC value as torch.Tensor
        """
        # mask = ((real != 0) & (prediction != 0)) & mask   # remove zero values in real tensor
        prediction = prediction[mask]
        real = real[mask]
        numerator = (prediction - prediction.mean()) * (real - real.mean())
        denominator = (prediction - prediction.mean()).pow(2).sum().sqrt() * (real - real.mean()).pow(2).sum().sqrt()

        return (numerator / denominator).sum() / mask.sum()

    def reset(self):
        """
        Reset internal accumulators.

        Initializes counts and sums for x, y, x^2, y^2, and xy.
        """
        self.total_samples = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, prediction: torch.Tensor, real: torch.Tensor, mask: torch.Tensor = None):
        """
        Accumulate statistics from a new batch.

        :param prediction: Predicted values (1d, 2d, or 3d tensor)
        :param real:       Real values (same shape as prediction)
        :param mask:       Optional mask tensor (same shape) to select valid entries
        """
        # Apply mask if provided
        if mask is not None:
            prediction = prediction[mask]
            real = real[mask]

        # Update counters and sums
        self.sum_x += prediction.sum().detach()
        self.sum_y += real.sum().detach()
        self.sum_x2 += (prediction * prediction).sum().detach()
        self.sum_y2 += (real * real).sum().detach()
        self.sum_xy += (prediction * real).sum().detach()
        self.total_samples += prediction.numel()

    def compute(self) -> torch.Tensor:
        """
        Compute the Pearson Correlation Coefficient using accumulated sums.

        :return: PCC value as torch.Tensor
        """
        if self.total_samples == 0:
            # No samples, return zero correlation
            return torch.tensor(0.0)

        # Compute covariance and variances
        cov_xy = self.sum_xy - (self.sum_x * self.sum_y) / self.total_samples
        var_x = self.sum_x2 - (self.sum_x ** 2) / self.total_samples
        var_y = self.sum_y2 - (self.sum_y ** 2) / self.total_samples

        denominator = (var_x * var_y).sqrt()

        return cov_xy / denominator
