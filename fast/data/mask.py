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

        random_mask = torch.rand(mask.shape, device=mask.device) < self.keep_ratio
        random_mask &= mask

        return random_mask


class BlockMask(AbstractMask):
    """
    Faster BlockMask: generates continuous blocks along time axis with vectorized operations.
    """

    def __init__(self, block_size: int, keep_ratio: float):
        assert 0 < block_size, "Block size must be positive."
        assert 0.0 <= keep_ratio <= 1.0, "keep_ratio must be in [0.0, 1.0]"
        self.block_size = block_size
        self.keep_ratio = keep_ratio

    def generate(self, mask: torch.Tensor):
        device = mask.device
        shape = mask.shape
        block_mask = torch.zeros_like(mask, dtype=torch.bool)

        if len(shape) == 2:  # (T, D)
            T, D = shape
            total_true_needed = int(self.keep_ratio * T * D)
            blocks_needed = max(1, total_true_needed // (self.block_size * D))

            starts = torch.randint(0, T - self.block_size + 1, (blocks_needed,), device=device)
            for start in starts:
                block_mask[start:start + self.block_size, :] = True

        elif len(shape) == 3:  # (B, T, D)
            B, T, D = shape
            total_true_needed = int(self.keep_ratio * T * D)
            blocks_needed = max(1, total_true_needed // (self.block_size * D))

            # 向量化生成所有 batch 的起始位置
            starts = torch.randint(0, T - self.block_size + 1, (B, blocks_needed), device=device)

            # 用广播一次性赋值
            t_idx = torch.arange(self.block_size, device=device).view(1, 1, -1)
            starts_expanded = starts.unsqueeze(-1) + t_idx  # (B, blocks_needed, block_size)
            starts_expanded = starts_expanded.clamp(max=T - 1)

            for b in range(B):
                block_mask[b, starts_expanded[b].reshape(-1), :] = True
        else:
            raise ValueError("Shape must be (T, D) or (B, T, D).")

        return block_mask & mask


class VariableMask(AbstractMask):
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
