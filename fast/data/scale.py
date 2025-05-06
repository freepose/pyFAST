#!/usr/bin/env python
# encoding: utf-8

"""
    Data normalization techniques.

    (1) Scaler applies to the whole (trainable part of) time series.

    (2) Instance scaler normalized batch input data of a model.

    (3) The scalers work with mask tensor to ignore missing values or padding values of a real (target) tensor.
"""

import copy
from typing import Tuple, List

import torch
import torch.nn as nn

from abc import abstractmethod, ABC


class AbstractScale(ABC):
    """
        AbstractScale class.
    """

    @abstractmethod
    def fit(self, x, x_mask=None):
        return self

    @abstractmethod
    def transform(self, x, x_mask=None) -> torch.Tensor:
        return x

    def fit_transform(self, x, x_mask=None) -> torch.Tensor:
        self.fit(x, x_mask)
        normalized_x = self.transform(x, x_mask)
        return normalized_x

    @abstractmethod
    def inverse_transform(self, x, x_mask=None) -> torch.Tensor:
        return x


class MinMaxScale(AbstractScale):
    """
        Min-Max normalization. Support high-dimensional data.
        :param feature_range: the range of normalized data. Default is ``[0, 1]``.
    """

    def __init__(self, feature_range: tuple or list = (0.0, 1.0)):
        self.feature_range = feature_range

        self.min = None
        self.max = None
        self.range = None
        self.given_range = self.feature_range[1] - self.feature_range[0]

    def fit(self, x: torch.tensor, x_mask: torch.tensor = None):
        """ x shape is [..., num_features]. """
        assert x.ndim > 1, "The input tensor must be at least 2D."

        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

        self.range = self.max - self.min
        return self

    def transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        normalized_x = (x - self.min) / self.range * self.given_range + self.feature_range[0]
        return normalized_x

    def inverse_transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        recovered_x = (x - self.feature_range[0]) * self.range / self.given_range + self.min
        return recovered_x


class MaxScale(AbstractScale):
    """ Max normalization (to be tested). sklearn: MaxAbsScale. NOTE: This seems no effects on input dataset. """

    def __init__(self):
        self.max = None

    def fit(self, x: torch.tensor, x_mask: torch.tensor = None):
        self.max = x.max(dim=0).values
        return self

    def transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        normalized_x = x / self.max
        return normalized_x

    def inverse_transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        recovered_x = x * self.max
        return recovered_x


class MeanScale(AbstractScale):
    """
        Mean normalization. Each value is divided by the mean value of each column.
    """

    def __init__(self):
        self.mean = None

    def fit(self, x: torch.tensor, x_mask: torch.tensor = None):
        self.mean = x.mean(dim=0)
        return self

    def transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        normalized_x = x / self.mean
        return normalized_x

    def inverse_transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        recovered_x = x * self.mean
        return recovered_x


class StandardScale(AbstractScale):
    """Standard normalization (a.k.a, z-score)."""

    def __init__(self):
        self.miu = None
        self.sigma = None

    def fit(self, x: torch.tensor, x_mask: torch.tensor = None):
        self.miu = x.mean(dim=0)
        self.sigma = x.var(dim=0).sqrt()
        return self

    def transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        normalized_x = (x - self.miu) / self.sigma
        return normalized_x

    def inverse_transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        recovered_x = x * self.sigma + self.miu
        return recovered_x


class LogScale(AbstractScale):
    """
        Logarithmic normalization.
    """

    def fit(self, x: torch.tensor, x_mask: torch.tensor = None):
        return self

    def transform(self, x: torch.tensor, x_mask: torch.tensor = None):
        return torch.log(x + 1)

    def inverse_transform(self, x: torch.tensor, x_mask: torch.tensor = None):
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
        self.fit(x)
        normalized_x = self.transform(x)
        return normalized_x

    @abstractmethod
    def inverse_transform(self, x, mask=None):
        return x


class InstanceStandardScale(InstanceScale):
    """
        Standard normalization on batch instances in forward feedback. A.k.a. ReVIN.

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

    def fit(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.miu = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.sigma = (x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.epsilon).sqrt().detach()
        return self

    def transform(self, x, mask=None):
        """ Normalization """
        t = (x - self.miu) / self.sigma  # -> [batch_size, input_window_size, num_features]
        if self.num_features > 0:
            t = (t * self.affine_weight) + self.affine_bias
        return t

    def inverse_transform(self, x, mask=None):
        """ De-normalize.  """
        if self.num_features > 0:
            x = (x - self.affine_bias) / (self.affine_weight + self.epsilon ** 2)
        inv = x * self.sigma + self.miu  # -> [batch_size, output_window_size, num_features]
        return inv


"""
    Multiple time series normalization techniques.
"""


def scale_several_time_series(scaler: AbstractScale,
                              ts: torch.Tensor or List[torch.Tensor] or Tuple[torch.Tensor],
                              mask: torch.Tensor or List[torch.Tensor] or Tuple[torch.Tensor] = None) -> AbstractScale:
    """
        Scale the datasets, and return the scaler.
        :param scaler: the scaler to be used. A deep copy will apply to the scaler.
        :param ts: time series tenor, or a list of time series tensors.
        :param mask: mask tensor or a list of mask tensors.
        :return: the fitted scaler.
    """
    if mask is not None:
        if isinstance(ts, torch.Tensor):
            assert ts.shape == mask.shape, 'ts and mask must have the same shape.'
        elif isinstance(ts, (List, Tuple)):
            assert len(ts) == len(mask.shape), 'ts and mask must have the same number of data sources.'

    ts_tensor, mask_tensor = ts, mask
    if isinstance(ts, (List, Tuple)):
        ts_tensor = torch.cat(ts, dim=0)
        if mask is not None:
            mask_tensor = torch.cat(mask, dim=0)

    scaler = copy.deepcopy(scaler).fit(ts_tensor, mask_tensor)

    return scaler
