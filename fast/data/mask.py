#!/usr/bin/env python
# encoding: utf-8

"""

    Mask strategy for Datasets.

    The word 'masker' is used to avoid confusion with 'mask' (indicator), and is used as a suffix of class names.


    (1) Static mask strategy: generate masks based on the original mask provided by the datasets (T, D).

    (2) Dynamic mask strategy: generate masks based on the original mask provided by the datasets (B, T, D).

    The key principles are:

    (1) Both dynamic and static masking strategies should be based on original mask, which is provided by the datasets.

    (2) The loss mask is (original mask - (original mask & generated mask)),
        which means that the loss is only calculated on the points that are originally valid
        but are masked by the generated mask.

    (3) For self-supervised pre-training, (1) and (2) is the foundation of mask modeling.

"""

import torch

from abc import abstractmethod, ABC
from typing import Union, List, Tuple
from .smt_dataset import TensorSequence


class AbstractMasker(ABC):
    """
        Abstract base class for time series masking strategies.

        This class defines the interface for generating masks that can be applied
        to time series data for various purposes such as data augmentation,
        missing value simulation, or self-supervised learning.

        The mask generation should follow these principles:
        1. Respect the original data validity mask
        2. Support both static (T, D) and dynamic (B, T, D) masking
        3. Enable composable masking strategies
    """

    @abstractmethod
    def generate(self, mask: torch.Tensor) -> torch.Tensor:
        """
            Apply the mask strategy to the dataset.

            If ``mask`` is None, a new mask will be generated based on the provided shape
            If ``mask`` is not None, it will be used to combine with the generated mask (**&** operation).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class RandomMasker(AbstractMasker):
    """
        ``RandomMask`` class generates a random point mask based on the provided ratio and shape.

        :param keep_ratio: the ratio of True values, range [0.0, 1.0].
    """

    def __init__(self, keep_ratio: float):
        assert 0.0 <= keep_ratio <= 1.0, "Ratio must be between 0.0 and 1.0"
        self.keep_ratio = keep_ratio

    def generate(self, mask: torch.Tensor):
        """
            Generate a random mask with the specified probability.

            :param mask: An existing mask to be combined with the generated mask.
                         True means keep, False means mask.
            :return: A boolean mask tensor of the same shape as the input
        """

        # if mask is None:
        #     raise ValueError("Input mask cannot be None")
        # if mask.dtype != torch.bool:
        #     raise TypeError(f"Expected mask to be bool tensor, got {mask.dtype}")

        random_mask = torch.rand(mask.shape, device=mask.device) < self.keep_ratio
        random_mask &= mask

        return random_mask


class BlockMasker(AbstractMasker):
    """
    BlockMask: generates continuous blocks along time-axis using vectorized ops.
    """

    def __init__(self, block_size: int, keep_ratio: float):
        assert block_size > 0, "block_size must be > 0"
        assert 0.0 <= keep_ratio <= 1.0, "keep_ratio must be in [0, 1]"
        self.block_size = block_size
        self.keep_ratio = keep_ratio

    def generate(self, mask: torch.Tensor) -> torch.Tensor:
        device = mask.device
        shape = mask.shape
        block_mask = torch.zeros_like(mask, dtype=torch.bool)

        if len(shape) == 2:  # (T, D)
            T, D = shape
            num_blocks = max(1, int((T * self.keep_ratio) // self.block_size))
            starts = torch.randint(0, T - self.block_size + 1, (num_blocks,), device=device)
            time_idx = (starts[:, None] + torch.arange(self.block_size, device=device)).flatten()
            block_mask[time_idx.clamp(max=T - 1), :] = True

        elif len(shape) == 3:  # (B, T, D)
            B, T, D = shape
            num_blocks = max(1, int((T * self.keep_ratio) // self.block_size))
            starts = torch.randint(0, T - self.block_size + 1, (B, num_blocks), device=device)
            time_idx = (starts[..., None] + torch.arange(self.block_size, device=device)).clamp(max=T - 1)
            b_idx = torch.arange(B, device=device)[:, None, None].expand_as(time_idx)
            block_mask[b_idx.reshape(-1), time_idx.reshape(-1), :] = True

        else:
            raise ValueError("mask shape must be (T, D) or (B, T, D)")

        return block_mask & mask


class VariableMasker(AbstractMasker):
    """
    Faster VariableMask: masks entire feature columns (variables) with vectorized operations.
    """

    def __init__(self, keep_ratio: float):
        assert 0.0 <= keep_ratio <= 1.0, "keep_ratio must be in [0.0, 1.0]"
        self.keep_ratio = keep_ratio

    def generate(self, mask: torch.Tensor):
        device = mask.device
        shape = mask.shape
        variable_mask = torch.zeros_like(mask, dtype=torch.bool)

        if len(shape) == 2:  # (T, D)
            T, D = shape
            vars_to_keep = max(1, int(D * self.keep_ratio))
            selected_vars = torch.randperm(D, device=device)[:vars_to_keep]
            variable_mask[:, selected_vars] = True

        elif len(shape) == 3:  # (B, T, D)
            B, T, D = shape
            vars_to_keep = max(1, int(D * self.keep_ratio))

            all_selected = torch.stack([torch.randperm(D, device=device)[:vars_to_keep] for _ in range(B)])

            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, vars_to_keep)
            variable_mask[b_idx, :, all_selected] = True
        else:
            raise ValueError("Shape must be (T, D) or (B, T, D).")

        return variable_mask & mask


def masker_generate(masker: AbstractMasker,
                    ts_mask: Union[torch.Tensor, TensorSequence]) -> Union[torch.Tensor, TensorSequence]:
    """
        Generate a mask using the provided mask instance.

        :param masker: An instance of AbstractMask or its subclasses.
        :param ts_mask: The original mask tensor or tensor list to combine with the generated mask.

        :return: A boolean tensor representing the generated mask.
    """

    generated_ts_mask = None
    if isinstance(ts_mask, torch.Tensor):
        generated_ts_mask = masker.generate(ts_mask)
    elif isinstance(ts_mask, List) or isinstance(ts_mask, Tuple):
        generated_ts_mask = [masker.generate(mask) for mask in ts_mask]

    return generated_ts_mask
