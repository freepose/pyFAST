#!/usr/bin/env python
# encoding: utf-8

__version__ = '0.0.0'


def initial_seed(seed: int = 10):
    """ Fix seed for random number generator. """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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
