#!/usr/bin/env python
# encoding: utf-8

"""
    Activation function selector.
"""

import torch.nn as nn

from typing import Literal, Type

ActivationName = Literal['linear', 'relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'sin']


class Sin(nn.Module):
    """
        Sine activation function.
    """

    def forward(self, x):
        out = x.sin()
        return out


def get_activation_cls(activation: ActivationName = 'linear') -> Type[nn.Module]:
    """
        Get activation function class by name.
        :param activation: activation function name,
                           can be one in ['linear', 'relu', 'gelu', 'elu', 'selu', 'tanh', 'sigmoid', 'silu', 'sin'].
        :return: activation function class.
    """
    activation_dict = {
        'linear': nn.Identity,
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'elu': nn.ELU,
        'selu': nn.SELU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'silu': nn.SiLU,
        'sin': Sin,
    }

    activation_fn = activation_dict.get(activation, activation_dict['linear'])
    return activation_fn
