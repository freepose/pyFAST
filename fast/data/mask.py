#!/usr/bin/env python
# encoding: utf-8

"""

    Mask strategy for Datasets. These classes apply static masking strategies on loading datasets,
    and dynamic masking strategies on the fly during training.

"""

import abc
import torch

from typing import Tuple, Union, List

TensorShape = Union[int, Tuple[int, ...], List[int]]


def random_point_mask(X, p=0.2):
    # X: (batch, seq_len, feature)
    observed_mask = ~torch.isnan(X)
    dynamic_mask = (torch.rand_like(X) > p)
    final_mask = observed_mask & dynamic_mask
    return final_mask


class AbstractMask(abc.ABC):
    """
        ``AbstractMask`` class.
    """

    def __init__(self):
        pass

    def __call__(self, raio: float, shape: TensorShape, mask: torch.Tensor = None):
        return self.generate(raio, shape, mask)

    def generate(self, raio: float, shape: TensorShape, mask: torch.Tensor = None):
        """
            Apply the mask strategy to the dataset.

            If ``mask`` is None, a new mask will be generated based on the provided shape
            If ``mask`` is not None, it will be used to overwrite the generated mask.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

