#!/usr/bin/env python
# encoding: utf-8

"""
    Data normalization techniques.

    (1) Scaler applies to the whole (trainable part of) time series.

    (2) Instance scaler normalized batch input data of a model.

    (3) The scalers work with mask tensor to ignore missing values or padding values of a real (target) tensor.
"""

import copy
import torch
import torch.nn as nn

from typing import Tuple, List, Union, Optional
from abc import abstractmethod, ABC
from .smt_dataset import TensorSequence


class AbstractScale(ABC):
    """
        AbstractScale class.
    """

    @abstractmethod
    def fit(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> 'AbstractScale':
        """ Fit the scaler to the data and return self for method chaining. """
        return self

    @abstractmethod
    def transform(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Transform data using fitted parameters with zero-division protection. """
        return x

    def fit_transform(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.fit(x, x_mask)
        normalized_x = self.transform(x, x_mask)
        return normalized_x

    @abstractmethod
    def inverse_transform(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class MinMaxScale(AbstractScale):
    """
        Min-Max normalization. Support high-dimensional data.
        :param feature_range: the range of normalized data. Default is ``[0, 1]``.
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range

        self.min = None
        self.max = None
        self.range = None
        self.given_range = self.feature_range[1] - self.feature_range[0]

    def fit(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        """ x shape is [..., num_features]. """
        assert x.ndim > 1, "The input tensor must be at least 2D."

        nan_mask = torch.isnan(x) if x_mask is None else ~x_mask  # -> [seq_len, num_features]

        self.max = torch.where(nan_mask, torch.tensor(-float('inf')), x).max(dim=0).values  # -> [..., num_features]
        self.min = torch.where(nan_mask, torch.tensor(float('inf')), x).min(dim=0).values  # -> [..., num_features]

        self.range = self.max - self.min

        return self

    def transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        normalized_x = (x - self.min) / self.range * self.given_range + self.feature_range[0]
        return normalized_x

    def inverse_transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        recovered_x = (x - self.feature_range[0]) * self.range / self.given_range + self.min
        return recovered_x


class MaxScale(AbstractScale):
    """ Max normalization (to be tested). sklearn: MaxAbsScale. NOTE: This seems no effects on input dataset. """

    def __init__(self):
        self.max = None

    def fit(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        """ x shape is [..., num_features]. """
        assert x.ndim > 1, "The input tensor must be at least 2D."

        nan_mask = torch.isnan(x) if x_mask is None else ~x_mask  # -> [seq_len, num_features]
        self.max = torch.where(nan_mask, torch.tensor(-float('inf')), x).max(dim=0).values  # -> [seq_len, num_features]

        return self

    def transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        normalized_x = x / self.max
        return normalized_x

    def inverse_transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        recovered_x = x * self.max
        return recovered_x


class MeanScale(AbstractScale):
    """
        Mean normalization. Each value is divided by the mean value of each column.
    """

    def __init__(self):
        self.mean = None

    def fit(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        """ x shape is [..., num_features]. """
        assert x.ndim > 1, "The input tensor must be at least 2D."

        if x_mask is not None:
            nan_mask = ~x_mask
            valid_counts = (~nan_mask).sum(dim=0).float()
            x_copy = x.clone()
            x_copy[nan_mask] = 0
            self.mean = x_copy.sum(dim=0) / valid_counts
            return self

        self.mean = x.mean(dim=0)
        return self

    def transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        normalized_x = x / self.mean
        return normalized_x

    def inverse_transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        recovered_x = x * self.mean
        return recovered_x


class StandardScale(AbstractScale):
    """Standard normalization (a.k.a, z-score)."""

    def __init__(self):
        self.miu = None
        self.sigma = None

    def fit(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        """ x shape is [..., num_features]. """

        assert x.ndim > 1, "The input tensor must be at least 2D."

        if x_mask is not None:
            nan_mask = ~x_mask
            valid_counts = (~nan_mask).sum(dim=0).float()
            x_copy = x.clone()
            x_copy[nan_mask] = 0
            self.miu = x_copy.sum(dim=0) / valid_counts
            self.sigma = (x_copy.var(dim=0, unbiased=False) + 1e-5).sqrt()
            return self

        self.miu = x.mean(dim=0)
        self.sigma = x.var(dim=0).sqrt()
        return self

    def transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        normalized_x = (x - self.miu) / self.sigma
        return normalized_x

    def inverse_transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        recovered_x = x * self.sigma + self.miu
        return recovered_x


class LogScale(AbstractScale):
    """
        Logarithmic normalization.
    """

    def fit(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        return self

    def transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        return torch.log(x + 1)

    def inverse_transform(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        return torch.exp(x) - 1


"""
    Instance normalization techniques.
"""


class InstanceScale(nn.Module):
    """ Base class of instance normalization. """

    def __init__(self):
        super(InstanceScale, self).__init__()

    @abstractmethod
    def fit(self, x, mask=None):
        return self

    @abstractmethod
    def transform(self, x, mask=None):
        return x

    def fit_transform(self, x, mask=None):
        self.fit(x, mask)
        normalized_x = self.transform(x, mask)
        return normalized_x

    @abstractmethod
    def inverse_transform(self, x, mask=None):
        return x


class InstanceStandardScale(InstanceScale):
    """
        Standard normalization on batch instances in forward feedback. A.k.a. **ReVIN**.

        Kim T, Kim J, Tae Y, et al.
        Reversible instance normalization for accurate time-series forecasting against distribution shift
        ICLR 2021.

        :param num_features: if num_features > 0, then use affine parameters, else not.
        :param epsilon: the default value is 1e-5.
    """

    def __init__(self, num_features: int = -1, epsilon: float = 1e-5):
        super(InstanceScale, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon

        self.miu = None
        self.sigma = None

        if self.num_features > 0:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def fit(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
            Fit the instance normalization parameters.
            :param x: the input tensor, shape is [..., num_features].
            :param mask: the mask tensor, shape is [..., num_features].
            :return: self, the fitted scaler.
        """

        assert x.ndim > 2, "The input tensor must be at least 3D."

        dim2reduce = tuple(range(1, x.ndim-1))

        if mask is not None:
            assert x.shape == mask.shape, 'x and mask must have the same shape.'
            nan_mask = ~mask
            valid_counts = (~nan_mask).sum(dim=dim2reduce, keepdim=True).float()

            # if valid_counts.min() == 0:
            #     raise ValueError("There are some features with all NaN values in the input tensor.")

            x_copy = x.clone()
            x_copy[nan_mask] = 0

            self.miu = x_copy.sum(dim=dim2reduce, keepdim=True) / (valid_counts + self.epsilon) # avoid division by zero
            self.sigma = (x_copy.var(dim=dim2reduce, unbiased=False, keepdim=True) + self.epsilon).sqrt()

            if self.num_features > 0:
                self.affine_weight.data.fill_(1.0)
                self.affine_bias.data.fill_(0.0)

            return self

        self.miu = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.sigma = (x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.epsilon).sqrt().detach()
        return self

    def transform(self, x: torch.Tensor, mask: torch.Tensor = None):
        """ Normalization """
        t = (x - self.miu) / self.sigma  # -> [..., num_features]
        if self.num_features > 0:
            t = (t * self.affine_weight) + self.affine_bias
        return t

    def inverse_transform(self, x: torch.Tensor, mask: torch.Tensor = None):
        """ De-normalize.  """
        if self.num_features > 0:
            x = (x - self.affine_bias) / (self.affine_weight + self.epsilon ** 2)
        inv = x * self.sigma + self.miu  # -> [..., num_features]
        return inv


"""
    Multiple time series normalization techniques.
"""


def scaler_fit(scaler: AbstractScale,
               ts: Union[torch.Tensor, TensorSequence],
               mask: Union[torch.Tensor, TensorSequence] = None) -> AbstractScale:
    """
        Fit the scaler to the time series data, without modifying the original time series data.

        :param scaler: the scaler to be used. A deep copy will apply to the scaler.
        :param ts: time series tenor, or a list of time series tensors.
        :param mask: mask tensor or a list of mask tensors.
        :return: the fitted scaler.
    """
    if mask is not None:
        if isinstance(ts, torch.Tensor):
            assert ts.shape == mask.shape, 'ts and mask must have the same shape.'
        elif isinstance(ts, (Tuple, List)):
            assert len(ts) == len(mask), 'ts and mask must have the same length.'

    ts_tensor, mask_tensor = ts, mask
    if isinstance(ts, (Tuple, List)):
        ts_tensor = torch.cat(ts, dim=0)
        if mask is not None:
            mask_tensor = torch.cat(mask, dim=0)

    scaler = copy.deepcopy(scaler).fit(ts_tensor, mask_tensor)

    return scaler


def scaler_transform(scaler: AbstractScale,
                     ts: Union[torch.Tensor, TensorSequence],
                     mask: Union[torch.Tensor, TensorSequence] = None) -> torch.Tensor:
    """
        Transform the time series data using the fitted scaler.

        :param scaler: the fitted scaler.
        :param ts: time series tenor, or a list of time series tensors.
        :param mask: mask tensor or a list of mask tensors.
        :return: the transformed time series tensor.
    """
    if mask is not None:
        if isinstance(ts, torch.Tensor):
            assert ts.shape == mask.shape, 'ts and mask must have the same shape.'
        elif isinstance(ts, (Tuple, List)):
            assert len(ts) == len(mask), 'ts and mask must have the same length.'

    normalized_ts = None
    if isinstance(ts, torch.Tensor):
        normalized_ts = scaler.transform(ts, mask)
    elif isinstance(ts, (Tuple, List)):
        normalized_ts = []
        for i in range(len(ts)):
            normalized_ts.append(scaler.transform(ts[i], mask[i] if mask is not None else None))

    return normalized_ts


"""
    Scale name to class. 
"""

available_scales = {
    'abstract': AbstractScale,
    'minmax': MinMaxScale,
    'max': MaxScale,
    'mean': MeanScale,
    'standard': StandardScale,
    'log': LogScale,
    'instance': InstanceScale,
    'instance_standard': InstanceStandardScale,  # a.k.a., ReVIN
}
