#!/usr/bin/env python
# encoding: utf-8

__version__ = '0.0.1'

import os, random, inspect
import numpy as np
import torch

from typing import Any, Dict, List, Tuple, Union, Callable


def initial_seed(seed: int = 10):
    """ Fix seed for random number generator. """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(preferred_device: str = 'cpu'):
    """
        Get the device with fallback support.
        :param preferred_device: preferred device. Values in ['cpu', 'cuda:0', 'mps']. Default: 'cpu'.
        :return: if preferred device is available, then return the preferred device,
                otherwise return the fallback device (i.e., cpu).
    """
    assert preferred_device in ('cpu', 'mps') or preferred_device.__contains__('cuda'), \
        'preferred_device must be in [cpu, mps] or cuda:0, cuda:1, ...'

    if preferred_device.__contains__('cuda'):
        cuda_devices = preferred_device.split(':')
        if len(cuda_devices) > 1:
            os.environ['CUDA_VISIBLE_DEVICES'] = preferred_device.split(':')[1]
        preferred_device = 'cuda'

    if 'cuda' in preferred_device and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        return torch.device('cpu')

    if preferred_device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS is not available. Falling back to CPU.")
        return torch.device('cpu')

    device = torch.device(preferred_device)

    return device


def get_kwargs(func: Callable, **given_kwargs) -> Dict[str, Any]:
    """
        Get the default arguments from the function signature, and **update** them with the given keyword arguments.

        :param func: function object.
        :param given_kwargs: given keyword arguments, which will **override** the default arguments.
        :return : dictionary of arguments.
    """
    signature = inspect.signature(func)

    new_kwargs = {
        param.name: param.default
        for param in signature.parameters.values()
        if param.default is not inspect.Parameter.empty
    }
    new_kwargs.update(given_kwargs)

    return new_kwargs


def get_common_kwargs(func: Callable, given_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
        Get common parameters between the function signature and .
        :param func: function object.
        :param given_kwargs: dictionary of given keyword arguments.
        :return: dictionary of common arguments.
    """
    signature = inspect.signature(func)
    common_arguments = {k: v for k, v in given_kwargs.items() if k in signature.parameters}

    return common_arguments
