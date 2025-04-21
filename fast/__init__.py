#!/usr/bin/env python
# encoding: utf-8

__version__ = '0.0.0'

import os
import random
import numpy as np
import torch

from typing import Literal


def initial_seed(seed: int = 10):
    """ Fix seed for random number generator. """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(preferred_device: Literal['cpu', 'cuda', 'mps'] = 'cpu', cuda_visible_devices: str = '-1'):
    """
        Get the device with fallback support.
        :param preferred_device: preferred device. Values in ['cpu', 'cuda', 'mps']. Default: 'cpu'.
        :param cuda_visible_devices: CUDA_VISIBLE_DEVICES.
        :return: if preferred device is available, then return the preferred device,
                otherwise return the fallback device (i.e., cpu).
    """
    if preferred_device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    device_dict = {
        'cpu': torch.device('cpu'),
        'cuda': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'mps': torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    }

    device = device_dict.get(preferred_device, torch.device('cpu'))

    return device


def get_common_params(func, params: dict) -> dict:
    """
        Get common parameters from the function signature.
        :param func: function object.
        :param params: dictionary of parameters.
        :return: dictionary of common parameters.
    """
    import inspect

    signature = inspect.signature(func)
    common_params = {k: v for k, v in params.items() if k in signature.parameters}
    return common_params
