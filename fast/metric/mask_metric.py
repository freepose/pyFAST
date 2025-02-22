#!/usr/bin/env python
# encoding: utf-8

"""
    Mask metrics supports the incomplete time series forecasting.
"""

import torch


def mask_mean_absolute_error(prediction: torch.tensor, real: torch.tensor, mask: torch.tensor):
    """
        Element-wise metrics.
        :param prediction:  Predicted values (1d, 2d, or 3d torch tensor)
        :param real:        Real values (1d, 2d, or 3d torch tensor)
        :param mask:        Mask indicator of real values (1d, 2d, or 3d torch tensor)
        :return: mask MAE
    """
    error = real[mask] - prediction[mask]
    numerator = error.abs().sum()
    denominator = mask.sum()

    return numerator / denominator


def mask_mean_squared_error(prediction: torch.tensor, real: torch.tensor, mask: torch.tensor):
    """
        Element-wise metrics.
        :param prediction:  Predicted values (1d, 2d, or 3d torch tensor)
        :param real:        Real values (1d, 2d, or 3d torch tensor)
        :param mask:        Mask indicator of real values (1d, 2d, or 3d torch tensor)
        :return: mask MSE
    """
    error = real[mask] - prediction[mask]
    numerator = error.pow(2).sum()
    denominator = mask.sum()

    return numerator / denominator


def mask_root_mean_squared_error(prediction: torch.tensor, real: torch.tensor, mask: torch.tensor):
    """
        Element-wise metrics.
        :param prediction:  Predicted values (1d, 2d, or 3d torch tensor)
        :param real:        Real values (1d, 2d, or 3d torch tensor)
        :param mask:        Mask indicator of real values (1d, 2d, or 3d torch tensor)
        :return: mask RMSE
    """
    error = real[mask] - prediction[mask]
    numerator = error.pow(2).sum()
    denominator = mask.sum()

    return (numerator / denominator).sqrt()


def mask_mean_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor, mask: torch.tensor):
    """
        Element-wise metrics.
        :param prediction:  Predicted values (1d, 2d, or 3d torch tensor)
        :param real:        Real values (1d, 2d, or 3d torch tensor), note that zero values.
        :param mask:        Mask indicator of real values (1d, 2d, or 3d torch tensor)
        :return: mask RMSE
    """
    mask = (real != 0) & mask   # remove zero values in real tensor
    percentage_error = (real[mask] - prediction[mask]).abs() / real[mask].abs()

    numerator = percentage_error.sum()
    denominator = mask.sum()

    return numerator / denominator


def mask_symmetric_mean_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor, mask: torch.tensor):
    """
        Element-wise metrics.
        :param prediction:  Predicted values (1d, 2d, or 3d torch tensor)
        :param real:        Real values (1d, 2d, or 3d torch tensor)
        :param mask:        Mask indicator of real values (1d, 2d, or 3d torch tensor)
        :return: mask RMSE
    """
    # mask = ((real != 0) & (prediction != 0)) & mask   # remove zero values in real tensor
    percentage_error = (real[mask] - prediction[mask]).abs() / (real[mask].abs() + prediction[mask].abs())

    numerator = percentage_error.sum()
    denominator = mask.sum()

    return numerator / denominator
