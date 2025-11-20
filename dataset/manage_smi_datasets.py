#!/usr/bin/env python
# encoding: utf-8

"""

    The multisource time series dataset loading module. Each source is loaded from a CSV file.

    A dataset may consist of multiple sources, i.e, CSV files.
    All CSV files in a common datasets share the same data fields.

"""
import os, random, logging, zipfile

from typing import Literal, Tuple, List, Union, Dict, Any
from pathlib import Path

from dataset.manage_smt_datasets import smt_metadata

from fast.data import SMIDataset
from fast.data.processing import load_smi_datasets, retrieve_files_in_zip, SMIDatasetSequence


def prepare_smi_datasets(data_root: str,
                         dataset_name: str,
                         window_size: int = 24,
                         stride: int = 24,
                         split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                         split_strategy: Literal['intra', 'inter'] = None,
                         device: Union[str, Literal['cpu', 'mps', 'cuda']] = 'cpu',
                         **task_kwargs: Dict[str, Any]) -> SMIDatasetSequence:
    """
        Prepare several ``SMIDataset`` datasets for training, validation, and testing.

        The default **float type** is ``float32``, you can change it in ``load_smi_datasets()`` to ``float64`` if needed .

        :param data_root: the ``time_series`` directory.
        :param dataset_name: the name of the dataset.
        :param window_size: the window size of each sample.
        :param stride: the stride between two consecutive windows.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                    (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                    The default is none, which means non-split.
        :param split_strategy: the strategy to split the dataset, 'intra' or 'inter'.
                                'intra' means to split the dataset into training, validation, and test sets
                                within each source (CSV file).
                                'inter' means to split the dataset into training, validation, and test sets
                                across all sources (CSV files).
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       the dataset device can be **different** to the model device.
        :param task_kwargs: task settings for the dataset.
            - ts: the type of time series, 'univariate' or 'multivariate'. Default is 'univariate'.
            - use_ex: whether to use exogenous time series. Default is False.
            - ex_ts_mask: whether to mask exogenous time series. Default is False.
            - bidirectional: whether to use bidirectional time series windows. Default is False.
            - dynamic_padding: whether to use dynamic padding for variable-length time series. Default is False.
                               len(time_series) % window_size != 0, then the last window will be padded to window_size.
            - shuffle: whether to shuffle the data files. Default is False (for debugging).
            - show_loading_progress: whether to show loading progress. Default is True.
            - max_loading_workers: the maximum number of workers for loading data. Default is None (auto).

        :return: a ``SMIDatasetSequence`` instance or several instances.
    """

    assert dataset_name in smt_metadata, \
        f"Dataset '{dataset_name}' not found in metadata. The dataset name should be one of {list(smt_metadata.keys())}."

    given_metadata = smt_metadata[dataset_name]
    paths = given_metadata['paths']  # Paths are a list of paths or csv files

    for i, path in enumerate(paths):  # Update 'data_root' in paths
        if isinstance(path, str):
            paths[i] = os.path.normpath(path.format(root=data_root))
        elif isinstance(path, (tuple, list)) and len(path) == 2:
            paths[i] = (os.path.normpath(path[0].format(root=data_root)), path[1])

    task_ts = task_kwargs.get('ts', 'univariate')
    task_use_ex = task_kwargs.get('use_ex', False)
    task_ex_mask = task_kwargs.get('ex_ts_mask', False)
    task_bidirectional = task_kwargs.get('bidirectional', False)
    task_dynamic_padding = task_kwargs.get('dynamic_padding', False)
    task_show_loading_progress = task_kwargs.get('show_loading_progress', True)
    task_max_loading_workers = task_kwargs.get('max_loading_workers', 1)

    task_shuffle = task_kwargs.get('shuffle', True)    # False: keep the file order for debugging

    variables = given_metadata['columns'].get(task_ts, None)
    if variables is None:
        raise ValueError(f"Task type '{task_ts}' not found in dataset '{dataset_name}' metadata.")

    ex_variables = given_metadata['columns'].get('exogenous', None) if task_use_ex else None



    # It also supports zipped file(s), where csv file(s) in a inner file of the zip file.
    filenames = []
    for path in paths:
        if isinstance(path, str):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f'Path not found: {path}')

            if path.is_file():
                filenames.append(path)
            elif path.is_dir():
                csv_files = list(path.glob('*.csv'))
                csv_files.sort()
                filenames.extend(csv_files)
        elif isinstance(path, (tuple, list)) and len(path) == 2:
            zip_filename, inner_path = path
            inner_csv_filenames = retrieve_files_in_zip(zip_filename, inner_path)
            inner_csv_filenames.sort()
            zip_ref = zipfile.ZipFile(zip_filename, 'r')
            zip_csv_filenames = [(zip_ref, name) for name in inner_csv_filenames]
            filenames.extend(zip_csv_filenames)

    if len(filenames) == 0:
        raise FileNotFoundError(f'No CSV files found in paths: {paths}')

    if task_shuffle:
        random.shuffle(filenames)   # shuffle filenames for better data distribution

    logging.getLogger().info('Loading (all or part of) {} files in {}'.format(len(filenames), paths))
    load_smi_args = {
        'filenames': filenames,
        'variables': variables,
        'ex_variables': ex_variables,
        'mask_ex_variables': task_ex_mask,
        'window_size': window_size,
        'stride': stride,
        'bidirectional': task_bidirectional,
        'dynamic_padding': task_dynamic_padding,
        'split_ratios': split_ratios,
        'split_strategy': split_strategy,
        'device': device,
        'show_loading_progress': task_show_loading_progress,
        'max_loading_workers': task_max_loading_workers,
    }
    smt_datasets = load_smi_datasets(**load_smi_args)

    return smt_datasets
