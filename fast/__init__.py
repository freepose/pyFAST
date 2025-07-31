#!/usr/bin/env python
# encoding: utf-8

__version__ = '0.0.1'

import os, random, sys, inspect, logging, platform, re
import numpy as np

import torch

from typing import Any, Dict, Union, Callable, TextIO


# Uncomment the following lines to enable multi-GPU support

# os.environ['KMP_DUPLICATE_OK'] = 'True'   # the cause reason maybe mixing use of 'pip3 install' and 'conda install'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def initial_seed(seed: int = 10):
    """
        Fix seed for random number generator. Commonly used experimental reproduction.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def initial_logger(file: str = 'stdout',
                   level: int = logging.INFO) -> logging.Logger:
    """
    Initialize logger to manage log output, either to console or to a file.

    :param file: output destination, either a file path or a stream (e.g., sys.stdout).
    :param level: logging level, e.g., logging.INFO.
    :return: configured logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    if file in ('stdout', 'stderr'):
        stream = sys.stdout if file == 'stdout' else sys.stderr
        handler = logging.StreamHandler(stream)
    else:
        normalized_path = os.path.normpath(file)
        if is_valid_path(normalized_path):
            dir_name = os.path.dirname(normalized_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            handler = logging.FileHandler(normalized_path, mode='w', encoding='utf-8')
        else:
            raise ValueError(f"Invalid file path: '{file}'")

    handler.setLevel(level)
    logger.addHandler(handler)

    return logger


def get_device(preferred_device: str = 'cpu'):
    """
        Get the device with fallback support.
        :param preferred_device: preferred device. Valid values: 'cpu', 'mps', 'cuda', 'cuda:0', ...
        :return: if preferred device is available, then return the preferred device,
                otherwise return the fallback device (i.e., cpu).
    """
    if preferred_device is None:
        return torch.device('cpu')

    assert preferred_device in ['cpu', 'mps'] or 'cuda' in preferred_device, \
        'preferred_device must be in [cpu, mps, cuda] or cuda:0, cuda:1, ..., cuda:N'

    if 'cuda' in preferred_device and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU.")
        return torch.device('cpu')

    if preferred_device == 'mps' and not torch.backends.mps.is_available():
        logging.getLogger().info("Warning: MPS is not available. Falling back to CPU.")
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


def is_valid_path(path):
    """
    Check if a file path is valid according to OS constraints.
    Handles filenames, relative paths, and absolute paths.

    :param path: The file path to check.
    :return: True if the path is valid, False otherwise.
    """
    # Check if empty
    if not path or path.isspace():
        return False

    # Get the filename part
    filename = os.path.basename(path)

    # If filename is empty (e.g., path ends with separator), check parent directory
    if not filename:
        path = os.path.dirname(path)
        if not path:
            return False
        filename = os.path.basename(path)

    # Windows forbidden filename characters: \ / : * ? " < > |
    # Unix/Linux has fewer forbidden chars, mainly /
    if platform.system() == 'Windows':
        forbidden_chars = r'[*?"<>|]'  # \ / : are allowed in paths
    else:
        forbidden_chars = ''  # In Unix, / is not allowed in filenames, but handled during normalization

    # Check for forbidden characters in filename
    if forbidden_chars and re.search(forbidden_chars, filename):
        return False

    # Check if it's a Windows reserved name
    if platform.system() == 'Windows':
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                          'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                          'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        name_without_ext = os.path.splitext(filename)[0].upper()
        if name_without_ext in reserved_names:
            return False

    # Check each component of the path
    dirs = []
    head, tail = os.path.split(path)
    while tail:
        dirs.append(tail)
        head, tail = os.path.split(head)
    if head:
        dirs.append(head)

    # Check total path length (Windows typically limits to 260 characters)
    if platform.system() == 'Windows' and len(path) > 260:
        return False

    return True
