#!/usr/bin/env python
# encoding: utf-8

"""
    Utility functions supporting the models.
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Dict, Literal, Union


def rolling_forecasting(model, given_x: torch.Tensor, steps: int = 30) -> torch.Tensor:
    """ rolling forecasting function. steps === output_window_size. """
    test_seq = given_x  # => [batch_size, input_window_size, input_size]

    # => [batch_size, output_window_size, input_size]
    predictions = torch.zeros(given_x.shape[0], steps, given_x.shape[2], device=given_x.device)

    for step in range(steps):
        y_hat = model(test_seq)  # => [batch_size, output_window_size, input_size]
        predictions[:, step, :] = y_hat.squeeze(1)  # => [batch_size, output_window_size, input_size]
        test_seq = torch.cat([test_seq[:, 1:, :], y_hat], dim=1)

    return predictions


def count_parameters(model: nn.Module, unit: Literal['k', 'm', 'g', 'auto'] = 'auto') -> Dict[
    str, Union[int, float, str]]:
    """
    Count parameter numbers of a given torch module.

    :param model: the model to count parameters.
    :param unit: the unit of parameters.
           'k' for KB, 'm' for MB, 'g' for GB, 'auto' for automatic selection.
    :return: a dictionary containing trainable params, total params and unit.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Automatic unit selection based on parameter count
    if unit == 'auto':
        if total_params >= 1024 * 1024 * 1024:  # More than 1 billion parameters
            unit = 'G'
        elif total_params >= 1024 * 1024:  # More than 1 million parameters
            unit = 'M'
        else:  # Less than 1 million parameters
            unit = 'K'
    else:
        unit = str.upper(unit)

    divisor = 1
    if unit == 'K':  # a kilo
        divisor = 1024
    elif unit == 'M':  # a million
        divisor = 1024 * 1024
    elif unit == 'G':  # a billion
        divisor = 1024 * 1024 * 1024

    return {'trainable': trainable_params / divisor, 'total': total_params / divisor, 'unit': unit}


def get_constants(model: nn.Module) -> dict:
    """
        :param model: model instance.
        :return: a dictionary of model constants.
    """
    constant_dict = {key: value for key, value in vars(model).items() if
                     isinstance(value, (int, float, str, bool, list, tuple))
                     and not key.startswith('_') and not key == 'training'}

    return constant_dict


def get_model_info(model: nn.Module, count_unit: Literal['k', 'm', 'g', 'auto'] = 'auto') -> str:
    """
        Get the model information in string format.
        :param model: model instance.
        :param count_unit: the unit of parameters.
           'k' for KB, 'm' for MB, 'g' for GB, 'auto' for automatic selection.
        :return: the string of model information.
    """

    name = model.__class__.__name__
    constants = get_constants(model)
    params_counts = count_parameters(model, count_unit)

    count_str = '{:.2f}/{:.2f}{}'.format(params_counts['trainable'], params_counts['total'], params_counts['unit'])
    params_dict = {**constants, 'trainable/total': count_str}

    params_str = ', '.join([f'{key}={value}' for key, value in params_dict.items()])
    params_str = '{}({})'.format(name, params_str)

    return params_str


def freeze_parameters(model: nn.Module) -> nn.Module:
    """
        Freeze the parameters of a given model.
        :param model: the model to freeze parameters.
        :return: the model with frozen parameters.
    """
    # Freeze the parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model


def covert_parameters(model: nn.Module, *args, **kwargs) -> nn.Module:
    """ covert float parameters to target types (dtype). """
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

    float_to_complex_dict = {torch.float16: torch.complex32,
                             torch.float32: torch.complex64,
                             torch.float64: torch.complex128}
    complex_dtype = float_to_complex_dict[dtype]

    def convert_float(t):
        if convert_to_format is not None and t.dim() in (4, 5):
            return t.to(device, dtype if t.is_floating_point() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, dtype if t.is_floating_point() else None, non_blocking)

    def convert_complex_float(t):
        if convert_to_format is not None and t.dim() in (4, 5):
            return t.to(device, complex_dtype if t.is_complex() else None,
                        non_blocking, memory_format=convert_to_format)
        return t.to(device, complex_dtype if t.is_complex() else None, non_blocking)

    ret = model._apply(convert_float)
    ret = ret._apply(convert_complex_float)

    return ret


def init_parameter_fn(parameter: torch.Tensor, fn_key, *args, **kwargs):
    if fn_key == "constant_":
        # nn.init.constant_(tensor, val)
        nn.init.constant_(parameter, *args, **kwargs)
    elif fn_key == "uniform_":
        # nn.init.uniform_(tensor, a=0, b=1)
        nn.init.uniform_(parameter, *args, **kwargs)
    elif fn_key == "normal_":
        # nn.init.normal_(tensor, mean=0, std=1)
        nn.init.normal_(parameter, *args, **kwargs)
    elif fn_key == "xavier_uniform_":
        # nn.init.xavier_uniform_(tensor, gain=1)
        nn.init.xavier_uniform_(parameter, *args, **kwargs)
    elif fn_key == "xavier_normal_":
        # nn.init.xavier_normal_(tensor, gain=1)
        nn.init.xavier_normal_(parameter, *args, **kwargs)
    elif fn_key == "kaiming_uniform_":
        # nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(parameter, *args, **kwargs)
    elif fn_key == "kaiming_normal":
        # nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(parameter, *args, **kwargs)
    elif fn_key == "orthogonal":
        # nn.init.orthogonal_(tensor, gain=1)
        nn.init.orthogonal_(parameter, *args, **kwargs)
    elif fn_key == "sparse":
        # nn.init.sparse_(tensor, sparsity, std=0.01)
        nn.init.sparse_(parameter, *args, **kwargs)
    elif fn_key == "zeros":
        # nn.init.zeros_(tensor)
        nn.init.zeros_(parameter)
    else:
        raise ValueError("Invalid fn_key in init_parameter_fn()")


def init_weights(module: nn.Module):
    """
        https://pytorch.org/docs/master/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module.apply
    """
    if isinstance(module, nn.Linear):
        init_parameter_fn(module.weight, "normal_", mean=0, std=0.01)
        if module.bias is not None:
            init_parameter_fn(module.bias, "constant_", 0)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
        init_parameter_fn(module.weight, "constant_", 1)
        if module.bias is not None:
            init_parameter_fn(module.bias, "constant_", 0)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        init_parameter_fn(module.weight, "xavier_uniform_")
        if module.bias is not None:
            init_parameter_fn(module.bias, "constant_", 0)
    elif isinstance(module, nn.GRU):
        std = 1.0 / np.sqrt(module.hidden_size) if module.hidden_size > 0 else 0
        for weight in module.parameters():
            nn.init.uniform_(weight, -std, std)
    elif isinstance(module, nn.Dropout):
        pass
    # else:
    #     module_name = type(module).__name__
    #     print(f"{module_name} has not been initialized weights.")


def to_string(*kwargs) -> str:
    """Several numbers to string."""
    _list = [str(kwargs[0])] + ['{:.6f}'.format(_t) for _t in kwargs[1:]]  # parameters to strings
    total = '\t'.join(_list)  # join these strings to another string
    return total
