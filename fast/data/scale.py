#!/usr/bin/env python
# encoding: utf-8
"""
    Data normalization techniques.
    Global scaler applies to the whole time series.
    Instance scaler normalized batch input data.
"""
import copy

import torch
import torch.nn as nn
from abc import abstractmethod, ABC


class Scale(object):
    """ Do nothing. """

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x, x_mask=None):
        return self

    @abstractmethod
    def transform(self, x, x_mask=None):
        return x

    @abstractmethod
    def fit_transform(self, x, x_mask=None):
        return x

    @abstractmethod
    def inverse_transform(self, x, x_mask=None):
        return x


class MinMaxScale(Scale):
    """
        Min-Max normalization. Support high-dimensional data.
        :param feature_range: the range of normalized data. Default is ``[0, 1]``.
    """

    def __init__(self, feature_range: tuple or list = (0.0, 1.0)):
        super(MinMaxScale, self).__init__()
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


class MaxScale(Scale):
    """ Max normalization (to be tested). sklearn: MaxAbsScale. NOTE: This seems no effects on input dataset. """

    def __init__(self):
        super(MaxScale, self).__init__()
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


class MeanScale(Scale):
    """Mean normalization. Each value is divided by the mean value of each column. """

    def __init__(self):
        super(MeanScale, self).__init__()
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


class StandardScale(Scale):
    """Standard normalization (a.k.a, z-score)."""

    def __init__(self):
        super(StandardScale, self).__init__()
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


class LogScale(Scale):
    """logarithmic normalization."""

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
        pass

    @abstractmethod
    def fit(self, x):
        return self

    @abstractmethod
    def transform(self, x):
        return x

    @abstractmethod
    def fit_transform(self, x):
        return x

    @abstractmethod
    def inverse_transform(self, x):
        return x


class InstanceStandardScale(InstanceScale):
    """
        Standard normalization on batch instances in forward feedback. A.k.a. ReVIN.

        Kim T, Kim J, Tae Y, et al.
        Reversible instance normalization for accurate time-series forecasting against distribution shift
        ICLR 2021.

        Nie Y, Nguyen N H, Sinthong P, et al.
        A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, ICLR 2023
        Link: https://arxiv.org/abs/2211.14730
        Official Code: https://github.com/yuqinie98/PatchTST

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

    def fit(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.miu = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.sigma = (x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.epsilon).sqrt().detach()
        return self

    def transform(self, x):
        """ Normalization """
        t = (x - self.miu) / self.sigma  # -> [batch_size, input_window_size, num_features]
        if self.num_features > 0:
            t = (t * self.affine_weight) + self.affine_bias
        return t

    def fit_transform(self, x):
        """ Fit and normalize. """
        self.fit(x)
        t = self.transform(x)
        return t

    def inverse_transform(self, x):
        """ De-normalize.  """
        if self.num_features > 0:
            x = (x - self.affine_bias) / (self.affine_weight + self.epsilon ** 2)
        inv = x * self.sigma + self.miu  # -> [batch_size, output_window_size, num_features]
        return inv

"""
    Mask normalization techniques.
"""


def time_series_scaler(ts: torch.Tensor or list[torch.Tensor], scaler: Scale()) -> Scale:
    """
        Scale the datasets.
        :param ts: the list of time series.
        :param scaler: the scaler.
        :return: the scaled time series.
    """
    if type(scaler) == type(Scale()):
        return Scale()

    scale_ts = ts
    if isinstance(ts, list):
        scale_ts = torch.cat(ts, dim=0)
    scaler = copy.deepcopy(scaler).fit(scale_ts)
    del scale_ts

    return scaler
