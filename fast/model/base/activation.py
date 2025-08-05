#!/usr/bin/env python
# encoding: utf-8

"""
    Activation function selector.
"""

import torch.nn as nn
from torch.nn import ReLU, GELU, ELU, SELU, Tanh, Sigmoid, SiLU, Identity


class Sin(nn.Module):
    """
        Sine activation function.
    """
    def forward(self, x):
        out = x.sin()
        return out


def get_activation_cls(activation: str = 'linear'):
    """
    Get the activation class.

    :param activation: The activation function to use.
    :param params: inplace = True or False.

    :return: The activation function.
    """
    activation_dict = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
        'linear': nn.Identity,
        'sin': Sin,
    }

    activation_fn = activation_dict.get(activation, activation_dict['linear'])
    return activation_fn
