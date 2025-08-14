#!/usr/bin/env python
# encoding: utf-8

"""

    Mask strategy for Datasets.

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


class AbstractMask(ABC):
    """
        AbstractMask class.
    """

    @abstractmethod
    def generate(self, mask: torch.Tensor):
        """
            Apply the mask strategy to the dataset.

            If ``mask`` is None, a new mask will be generated based on the provided shape
            If ``mask`` is not None, it will be used to combine with the generated mask (**&** operation).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class RandomMask(AbstractMask):
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

        random_mask = torch.rand(mask.shape) < self.keep_ratio
        random_mask = random_mask.to(device=mask.device) & mask

        return random_mask


class BlockMask(AbstractMask):
    """
        BlockMask class generates a block mask (continuous time spans) based on the provided ratio and shape.

        The block mask is also known as the span mask. It simulates missing data in continuous time spans.

        :param keep_ratio: the ratio of True values, range [0.0, 1.0].
        :param block_size: length of each continuous block to be masked along the time dimension.
    """

    def __init__(self, block_size: int, keep_ratio: float):
        assert 0 < block_size, "Block size must be a positive integer"
        assert 0.0 <= keep_ratio <= 1.0, "Ratio must be between 0.0 and 1.0"
        self.block_size = block_size
        self.keep_ratio = keep_ratio

    def generate(self, mask: torch.Tensor):
        """
            Generate a block mask with the specified probability.

            :param mask: An optional existing mask to be combined with the generated mask.
                         True means keep, False means mask.
            :return: A boolean mask tensor of the same shape as the input,
                     where True indicates the point is kept, and False indicates the point is masked.
        """

        block_mask = torch.zeros_like(mask)

        if len(mask.shape) == 2:  # (T, D)
            T, D = mask.shape
            num_blocks = max(1, int((T * self.keep_ratio) // self.block_size))
            for _ in range(num_blocks):
                start = torch.randint(0, T - self.block_size + 1, (1,)).item()
                block_mask[start:start + self.block_size, :] = True

        elif len(mask.shape) == 3:  # (B, T, D)
            B, T, D = mask.shape
            num_blocks = max(1, int((T * self.keep_ratio) // self.block_size))
            for b in range(B):
                for _ in range(num_blocks):
                    start = torch.randint(0, T - self.block_size + 1, (1,)).item()
                    block_mask[b, start:start + self.block_size, :] = True
        else:
            raise ValueError("Shape must be (T, D) or (B, T, D).")

        if mask is not None:
            block_mask = block_mask.to(device=mask.device) & mask

        return block_mask


class VariableMask(AbstractMask):
    """
        VariableMask class generates a variable-wise mask (entire feature columns).

        The variables mask simulates sensor/channel-level missing data.

        :param keep_ratio: the ratio of True values, range [0.0, 1.0].
        :param shape: The shape of the mask to be generated, usually (T, D) or (B, T, D).
    """

    def __init__(self, keep_ratio: float):
        assert 0.0 <= keep_ratio <= 1.0, "Ratio must be between 0.0 and 1.0"
        self.keep_ratio = keep_ratio

    def generate(self, mask: torch.Tensor):
        """
            Generate a feature-wise mask with the specified probability.

            :param mask: An optional existing mask to be combined with the generated mask.
                         True means keep, False means mask.
            :return: A boolean mask tensor of the same shape as the input,
                     where True indicates the point is kept, and False indicates the point is masked.
        """
        variable_mask = torch.zeros(mask.shape, dtype=torch.bool)

        if len(mask.shape) == 2:  # (T, D)
            T, D = mask.shape
            num_masked_variables = max(1, int(D * self.keep_ratio))
            selected_variables = torch.randperm(D)[:num_masked_variables]
            variable_mask[:, selected_variables] = True

        elif len(mask.shape) == 3:  # (B, T, D)
            B, T, D = mask.shape
            num_masked_variables = max(1, int(D * self.keep_ratio))
            for b in range(B):
                selected_variables = torch.randperm(D)[:num_masked_variables]
                variable_mask[b, :, selected_variables] = True
        else:
            raise ValueError("Shape must be (T, D) or (B, T, D).")

        variable_mask = variable_mask.to(device=mask.device) & mask

        return variable_mask
