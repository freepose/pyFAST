#!/usr/bin/env python
# encoding: utf-8

"""
    Loading tools for time series datasets.
"""

import os, sys, zipfile

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from typing import Literal, List, Tuple, Union, Dict, Any, Type, Optional

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from ... import get_device
from ..utils import collate_dict
from .. import SSTDataset, SMTDataset, SMCDataset, SMIDataset, SMIrDataset, AbstractSupervisedStrategy

SSTDatasetSequence = Union[SSTDataset, Tuple[SSTDataset, ...], List[SSTDataset]]
SMTDatasetSequence = Union[SMTDataset, Tuple[SMTDataset, ...], List[SMTDataset]]
SMIDatasetSequence = Union[SMIDataset, Tuple[SMIDataset, ...], List[SMIDataset]]
SMCDatasetSequence = Union[SMCDataset, Tuple[SMCDataset, ...], List[SMCDataset]]
SMIrDatasetSequence = Union[SMIrDataset, Tuple[SMIrDataset, ...], List[SMIrDataset]]


def retrieve_files_in_zip(zip_filename: str, inner_dir: str, ext: str = 'csv') -> List[str]:
    """
        Retrieve a list of CSV files from a specified directory within a ZIP archive.

        :param zip_filename: The path to the ZIP file.
        :param inner_dir: The directory inside the ZIP archive to search for CSV files.
        :param ext: The file extension to filter by (default is 'csv').

        :return: A list of CSV file names found in the specified directory.
    """

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        csv_files = [name for name in zip_ref.namelist() if name.startswith(inner_dir) and name.endswith(ext)]
        return csv_files


def read_csv_from_zip(zip_filename: str, inner_filename: str) -> pd.DataFrame:
    """
        Read a CSV file from within a ZIP archive and return it as a pandas DataFrame.

        :param zip_filename: The path to the ZIP file.
        :param inner_filename: The name of the CSV file inside the ZIP archive.
        :return: A pandas DataFrame containing the data from the CSV file.
    """
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        with zip_ref.open(inner_filename) as file:
            # df = pd.read_csv(file)
            df = pd.read_csv(file).assign(source_file=inner_filename)
            return df


def read_csv_from_zip_ref(zip_ref: zipfile.ZipFile, inner_filename: str) -> pd.DataFrame:
    """
        Read a CSV file from within a ZIP archive (using an existing ZipFile reference)
        and return it as a pandas DataFrame.

        :param zip_ref: An existing ZipFile reference.
        :param inner_filename: The name of the CSV file inside the ZIP archive.
        :return: A pandas DataFrame containing the data from the CSV file.
    """
    with zip_ref.open(inner_filename) as file:
        df = pd.read_csv(file).assign(source_file=inner_filename)
        return df


def load_sst_datasets(path: Union[str, Tuple[str, str], List[str]],
                      variables: List[str],
                      mask_variables: bool = False,
                      ex_variables: List[str] = None,
                      mask_ex_variables: bool = False,
                      ex2_variables: str = None,
                      input_window_size: int = 96,
                      output_window_size: int = 24,
                      horizon: int = 1,
                      stride: int = 1,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu') -> SSTDatasetSequence:
    """
        Load time series dataset from a **CSV** file (maybe, in a zipped file),
        transform time series data into supervised data,
        and split the dataset into several datasets.

        The default **float type** is ``float32``, you can change it to ``float64`` if needed.
        The default **device** is ``cpu``, you can change it to ``cuda`` or ``mps`` if needed.

        :param path: type str or tuple of list CSV filename, support CSV in a zip file.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param ex2_variables: names of the second exogenous variables. This is used for pre-known features,
                                such as time features, forecasted weather factors, etc.
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.
        :return: the (split) datasets as SSTDataset objects.
    """

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

    float_type = np.float32  # the default float type is ``float32``, you can change it to ``float64`` if needed
    device = get_device(device)

    df = pd.read_csv(path) if isinstance(path, str) else read_csv_from_zip(*path)

    sst_args = dict()

    target_df = df.loc[:, variables]
    target_array = target_df.values.astype(float_type)
    target_tensor = torch.tensor(target_array, device=device)
    sst_args['ts'] = target_tensor

    if mask_variables:
        mask_target_tensor = ~torch.isnan(target_tensor)
        sst_args['ts_mask'] = mask_target_tensor

    if ex_variables is not None:
        ex_df = df.loc[:, ex_variables]
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_args['ex_ts'] = ex_tensor

        if mask_ex_variables:
            mask_ex_tensor = ~torch.isnan(ex_tensor)
            sst_args['ex_ts_mask'] = mask_ex_tensor

    if ex2_variables is not None:
        ex_df = df.loc[:, ex2_variables]
        ex_array = ex_df.values.astype(float_type)
        ex_tensor = torch.tensor(ex_array, device=device)
        sst_args['ex_ts2'] = ex_tensor

    sst_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                     'horizon': horizon, 'stride': stride})

    if split_ratios is None:
        return SSTDataset(**sst_args)

    sst_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])
    for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
        if i > 0:
            sst_args.update({'stride': output_window_size})
        split_ds = SSTDataset(**sst_args).split(s, e, is_strict=False, mark='split_{}'.format(i))
        sst_datasets.append(split_ds)

    return sst_datasets


def get_smt_args(filenames: List[str],
                 variables: List[str], mask_variables: bool = False,
                 ex_variables: List[str] = None, mask_ex_variables: bool = False,
                 ex2_variables: List[str] = None,
                 float_type: np.dtype = np.float32,
                 device: torch.device = torch.device('cpu'),
                 show_progress: bool = True) -> Dict[str, Any]:
    """
        Get the arguments of **SMTDataset** dataset by reading several **CSV** files.

        :param filenames: list of CSV filenames.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param ex2_variables: names of the exogenous variables.
        :param float_type: the data type of the time series data, default is ``np.float32``.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
        :param show_progress: whether to show the progress bar.
                        Set to ``False`` to disable the progress bar.

        :return: a dictionary of arguments for **SMTDataset**.
    """

    smt_args = dict()
    smt_args['ts'] = []
    smt_args['ts_mask'] = [] if mask_variables else None
    smt_args['ex_ts'] = [] if ex_variables is not None else None
    smt_args['ex_ts_mask'] = [] if mask_ex_variables else None
    smt_args['ex_ts2'] = [] if ex2_variables is not None else None

    with tqdm(total=len(filenames), leave=False, file=sys.stdout, disable=not show_progress) as pbar:
        for filename in filenames:
            path_name = Path(filename)
            pbar.set_description('Loading {}'.format(path_name.parent.name + '/' + path_name.name))

            df = pd.read_csv(filename) if isinstance(filename, (str, Path)) else read_csv_from_zip_ref(*filename)

            target_df = df.loc[:, variables]
            target_array = target_df.values.astype(float_type)
            target_tensor = torch.tensor(target_array, device=device)
            smt_args['ts'].append(target_tensor)

            if mask_variables:
                mask_target_tensor = ~torch.isnan(target_tensor)
                smt_args['ts_mask'].append(mask_target_tensor)

            if ex_variables is not None:
                ex_df = df.loc[:, ex_variables]
                ex_array = ex_df.values.astype(float_type)
                ex_tensor = torch.tensor(ex_array, device=device)
                smt_args['ex_ts'].append(ex_tensor)

                if mask_ex_variables:
                    mask_ex_tensor = ~torch.isnan(ex_tensor)
                    smt_args['ex_ts_mask'].append(mask_ex_tensor)

            if ex2_variables is not None:
                ex_df = df.loc[:, ex_variables]
                ex_array = ex_df.values.astype(float_type)
                ex_tensor = torch.tensor(ex_array, device=device)
                smt_args['ex_ts2'].append(ex_tensor)

            pbar.update(1)

    return smt_args


def get_smt_args_parallel(filenames: List[Union[str, Tuple[str, str]]],
                          variables: List[str], mask_variables: bool = False,
                          ex_variables: List[str] = None, mask_ex_variables: bool = False,
                          ex2_variables: List[str] = None,
                          timepoint_variables: List[str] = None,
                          float_type: np.dtype = np.float32,
                          device: torch.device = torch.device('cpu'),
                          transpose: bool = False,
                          show_progress: bool = True,
                          max_workers: int = None) -> Dict[str, Any]:
    """
        Get the arguments of **SMTDataset** dataset by reading several **CSV** files.
        Uses multi-threading to process files concurrently.

        :param filenames: list of CSV filenames, support CSV in a zip file.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param ex2_variables: names of the exogenous variables.
        :param timepoint_variables: names of the time variables.
        :param float_type: the data type of the time series data, default is ``np.float32``.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
        :param transpose: whether to transpose the time series data while reading (in zipped) CSV files.
        :param show_progress: whether to show the progress bar.
                        Set to ``False`` to disable the progress bar in logging mode.
        :param max_workers: maximum number of threads, default is None (uses system default).

        :return: a dictionary of arguments for **SMTDataset**.
    """

    def process_file(filename: Union[str, Tuple[str, str]]) -> Dict[str, Any]:
        """Process a single CSV file and return tensors"""
        df = pd.read_csv(filename) if isinstance(filename, (Path, str)) else read_csv_from_zip_ref(*filename)

        if transpose:
            if 'source_file' in df.columns:
                df = df.drop(columns=['source_file'])

            new_df = df.T.copy()
            new_df.columns = new_df.iloc[0].astype(str)
            new_df = new_df.iloc[1:].reset_index(drop=True) # remove the first row (original columns)
            df = new_df.copy()

        ret_dict = {}

        # Process target variables
        target_df = df.loc[:, variables]
        target_array = target_df.values.astype(float_type)
        target_tensor = torch.tensor(target_array, device=device)
        ret_dict['ts'] = target_tensor

        # Process target mask
        if mask_variables:
            mask_target_array = ~np.isnan(target_array)
            mask_target_tensor = torch.tensor(mask_target_array, dtype=torch.bool, device=device)
            ret_dict['ts_mask'] = mask_target_tensor

        # Process exogenous variables
        if ex_variables is not None:
            ex_df = df.loc[:, ex_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            ret_dict['ex_ts'] = ex_tensor

            if mask_ex_variables:
                mask_ex_array = ~np.isnan(ex_array)
                mask_ex_tensor = torch.tensor(mask_ex_array, dtype=torch.bool, device=device)
                ret_dict['ex_ts_mask'] = mask_ex_tensor

        # Process second exogenous variables
        if ex2_variables is not None:
            ex_df = df.loc[:, ex2_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            ret_dict['ex_ts2'] = ex_tensor

        if timepoint_variables is not None:
            time_ts_df = df.loc[:, timepoint_variables]
            time_ts_array = time_ts_df.values.astype(float_type)
            time_ts_tensor = torch.tensor(time_ts_array, device=device)
            ret_dict['timepoint_ts'] = time_ts_tensor

        return ret_dict

    # Use thread_map to process files concurrently
    results = thread_map(
        process_file,
        filenames,
        max_workers=max_workers,
        desc="Loading files",
        disable=not show_progress,
        leave=False,
        file=sys.stdout
    )

    smt_args = collate_dict(results)    # Aggregate results: transform list of dicts to dict of leaf lists

    return smt_args


def load_smt_datasets(filenames: List[Union[str, Tuple[str, str], List[str]]],
                      variables: List[str],
                      mask_variables: bool = False,
                      ex_variables: List[str] = None,
                      mask_ex_variables: bool = False,
                      ex2_variables: List[str] = None,
                      input_window_size: int = 96,
                      output_window_size: int = 24,
                      horizon: int = 1,
                      stride: int = 1,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      split_strategy: Literal['intra', 'inter'] = 'intra',
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu',
                      show_loading_progress: bool = True,
                      max_loading_workers: int = None) -> SMTDatasetSequence:
    """
        Load **SMTDataset** from several **CSV** files, directories or a zip file,
        transform time series data into supervised data,
        and split the dataset into several datasets.

        :param filenames: list of CSV filenames, support CSV in a zip file.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param mask_variables: whether to mask the target variables. This uses for sparse_fusion time series
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion
            exogenous time series.
        :param ex2_variables: names of the second exogenous variables.
                            This is used for pre-known features, such as time features, forecasted weather factors, etc.
        :param input_window_size: input window size of the transformed supervised data
                            A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data
                            A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70 % for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param split_strategy: the strategy to split the dataset, should be 'intra' or 'inter', the default is 'inter'.
                                'intra' means splitting the dataset into several parts by the ratios,
                                'inter' means splitting the dataset by the filenames.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.
        :param show_loading_progress: whether to show the loading progress bar.
        :param max_loading_workers: maximum number of threads for loading data, default is None (uses system default).

        :return: the (split) datasets as ``SMTDataset`` or ``SMDDataset`` objects.
    """

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

        assert split_strategy in ('intra', 'inter'), \
            f"Invalid split strategy: {split_strategy}. The split strategy should be one of ['intra', 'inter']."

    float_type = np.float32
    device = get_device(device)
    show_pregress = show_loading_progress
    max_workers = max_loading_workers

    if split_ratios is None:
        smt_args = get_smt_args_parallel(filenames, variables, mask_variables, ex_variables, mask_ex_variables,
                                         ex2_variables, float_type=float_type, device=device, transpose=False,
                                         show_progress=show_pregress, max_workers=max_workers)
        smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                         'horizon': horizon, 'stride': stride})
        return SMTDataset(**smt_args)

    smt_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])

    if split_strategy == 'intra':
        smt_args = get_smt_args_parallel(filenames, variables, mask_variables, ex_variables, mask_ex_variables,
                                         ex2_variables, float_type=float_type, transpose=False,
                                         device=device, show_progress=show_pregress, max_workers=max_workers)
        smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                         'horizon': horizon, 'stride': stride})

        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            if i > 0:
                smt_args.update({'stride': output_window_size})
            split_ds = SMTDataset(**smt_args).split(s, e, is_strict=False, mark='intra_split_{}'.format(i))
            smt_datasets.append(split_ds)
    else:  # split_strategy == 'inter'
        filename_num = len(filenames)
        if filename_num < 2:
            raise ValueError(f'Invalid number of files: {filename_num}. '
                             f'The number of files should be at least 2 for "inter" split strategy.')

        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            start, end = int(filename_num * round(s, 10)), int(filename_num * round(e, 10))
            split_filenames = filenames[start:end]
            smt_args = get_smt_args_parallel(split_filenames, variables, mask_variables, ex_variables,
                                             mask_ex_variables, ex2_variables, float_type=float_type, transpose=False,
                                             device=device, show_progress=show_pregress, max_workers=max_workers)
            smt_args.update({'input_window_size': input_window_size, 'output_window_size': output_window_size,
                             'horizon': horizon, 'stride': stride})
            if i > 0:
                smt_args.update({'stride': output_window_size})

            split_dataset = SMTDataset(**smt_args, mark='inter_split_{}'.format(i))
            split_dataset.ratio = round(e - s, 10)
            smt_datasets.append(split_dataset)

    return smt_datasets


def get_smi_args_parallel(filenames: List[Union[str, Tuple[str, str]]],
                          variables: List[str],
                          ex_variables: List[str] = None,
                          mask_ex_variables: bool = False,
                          float_type: np.dtype = np.float32,
                          device: torch.device = torch.device('cpu'),
                          show_progress: bool = True,
                          max_workers: int = None) -> Dict[str, Any]:
    """
        Get the arguments of ** SMIDataset ** dataset by reading several ** CSV ** files.
        Uses multi-threading to process files concurrently.

        :param filenames: list of CSV filenames, support CSV in a zip file.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for sparse_fusion exogenous time series.
        :param float_type: the data type of the time series data, default is ``np.float32``.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
        :param show_progress: whether to show the progress bar.
                        Set to ``False`` to disable the progress bar in logging mode.
        :param max_workers: maximum number of threads, default is None (uses system default).

        :return: a dictionary of arguments for ** SMIDataset **.
    """

    def process_file(filename: Union[str, Tuple[str, str]]) -> Dict[str, Any]:
        """ Process a single CSV file and return tensors. """
        df = pd.read_csv(filename) if isinstance(filename, (Path, str)) else read_csv_from_zip_ref(*filename)
        result = {}

        # Process target variables
        target_df = df.loc[:, variables]
        target_array = target_df.values.astype(float_type)
        target_tensor = torch.tensor(target_array, device=device)
        result['ts'] = target_tensor

        # Process target mask
        mask_target_array = ~np.isnan(target_array)
        mask_target_tensor = torch.tensor(mask_target_array, dtype=torch.bool, device=device)
        result['ts_mask_input'] = mask_target_tensor
        result['ts_mask_output'] = mask_target_tensor.clone()

        # Process exogenous variables
        if ex_variables is not None:
            ex_df = df.loc[:, ex_variables]
            ex_array = ex_df.values.astype(float_type)
            ex_tensor = torch.tensor(ex_array, device=device)
            result['ex_ts'] = ex_tensor

            if mask_ex_variables:
                mask_ex_array = ~np.isnan(ex_array)
                mask_ex_tensor = torch.tensor(mask_ex_array, dtype=torch.bool, device=device)
                result['ex_ts_mask'] = mask_ex_tensor

        return result

    # Use thread_map to process files concurrently
    results = thread_map(
        process_file,
        filenames,
        max_workers=max_workers,
        desc="Loading files",
        disable=not show_progress,
        leave=False,
        file=sys.stdout
    )

    # Aggregate results
    smi_args = {
        'ts': [],
        'ts_mask_input': [],
        'ts_mask_output': [],
        'ex_ts': [] if ex_variables is not None else None,
        'ex_ts_mask': [] if ex_variables is not None and mask_ex_variables else None
    }

    for r in results:
        smi_args['ts'].append(r['ts'])
        smi_args['ts_mask_input'].append(r['ts_mask_input'])
        smi_args['ts_mask_output'].append(r['ts_mask_output'])
        if ex_variables is not None:
            smi_args['ex_ts'].append(r['ex_ts'])
            if mask_ex_variables:
                smi_args['ex_ts_mask'].append(r['ex_ts_mask'])

    return smi_args


def load_smi_datasets(filenames: List[Union[str, Tuple[str, str], List[str]]],
                      variables: List[str],
                      ex_variables: List[str] = None,
                      mask_ex_variables: bool = False,
                      window_size: int = 96,
                      stride: int = 1,
                      bidirectional: bool = False,
                      dynamic_padding: bool = False,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      split_strategy: Literal['intra', 'inter'] = None,
                      device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu',
                      show_loading_progress: bool = True,
                      max_loading_workers: int = None) -> SMIDatasetSequence:
    """
        Load **SMIDataset** from several **CSV** files, directories or a zip file,
        transform time series data into supervised data,
        and split the dataset into several datasets.

        :param filenames: list of CSV filenames, support CSV in a zip file.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for fusing exogenous time series.
        :param window_size: sliding window size.
        :param stride: the distance between two consecutive samples.
        :param bidirectional: whether to use backward window.
        :param dynamic_padding: whether to use dynamic padding window,
                                when the length of (remained/rest) time series is smaller than window size.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param split_strategy: the strategy to split the dataset, should be 'intra' or 'inter', the default is 'intra'.
                                'intra' means splitting the dataset into several parts by the ratios,
                                'inter' means splitting the dataset by the filenames.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.
        :param show_loading_progress: whether to show the loading progress bar.
        :param max_loading_workers: maximum number of threads for loading data, default is None (uses system default).

        :return: the (split) datasets as ``SMIDataset`` objects.
    """
    assert window_size > 0, f'Invalid window size: {window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

        assert split_strategy in ('intra', 'inter'), \
            f"Invalid split strategy: {split_strategy}. The split strategy should be one of ['intra', 'inter']."

    float_type = np.float32
    device = get_device(device)
    show_progress = show_loading_progress
    max_workers = max_loading_workers

    if split_ratios is None:
        smi_args = get_smi_args_parallel(filenames, variables, ex_variables, mask_ex_variables,
                                         float_type=float_type, device=device,
                                         show_progress=show_progress, max_workers=max_workers)
        smi_args.update({'window_size': window_size, 'stride': stride,
                         'bidirectional': bidirectional, 'dynamic_padding': dynamic_padding})
        return SMIDataset(**smi_args)

    smi_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])

    if split_strategy == 'intra':
        smi_args = get_smi_args_parallel(filenames, variables, ex_variables, mask_ex_variables,
                                         float_type=float_type, device=device,
                                         show_progress=show_progress, max_workers=max_workers)
        smi_args.update({'window_size': window_size, 'stride': stride,
                         'bidirectional': bidirectional, 'dynamic_padding': dynamic_padding})
        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            if i > 0:
                smi_args.update({'stride': window_size})
            split_ds = SMIDataset(**smi_args).split(s, e, mark='intra_split_{}'.format(i))
            smi_datasets.append(split_ds)
    else:  # split_strategy == 'inter'
        filename_num = len(filenames)
        if filename_num < 2:
            raise ValueError(f'Invalid number of files: {filename_num}. '
                             f'The number of files should be at least 2 for "inter" split strategy.')

        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            start, end = int(filename_num * round(s, 10)), int(filename_num * round(e, 10))
            split_filenames = filenames[start:end]
            smi_args = get_smi_args_parallel(split_filenames, variables, ex_variables, mask_ex_variables,
                                             float_type=float_type, device=device,
                                             show_progress=show_progress, max_workers=max_workers)
            smi_args.update({'window_size': window_size, 'stride': stride,
                             'bidirectional': bidirectional, 'dynamic_padding': dynamic_padding})
            if i > 0:
                smi_args.update({'stride': window_size})
            split_dataset = SMIDataset(**smi_args, mark='inter_split_{}'.format(i))
            split_dataset.ratio = round(e - s, 10)
            smi_datasets.append(split_dataset)

    return smi_datasets


def load_smc_datasets(filename: str,
                      input_window_size: int = 96,
                      output_window_size: int = 24,
                      horizon: int = 1,
                      stride: int = 1,
                      split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                      split_strategy: Literal['intra', 'inter'] = 'intra',
                      device: Union[Literal['cpu', 'mps', 'cuda'], str] = 'cpu',
                      global_ts_ids: Optional[Union[List[Any], torch.Tensor]] = None,
                      global_time_point_ids: Optional[Union[List[Any], torch.Tensor]] = None,
                      global_variable_ids: Optional[Union[List[Any], torch.Tensor]] = None) -> SMCDatasetSequence:
    """
        Load **SMCDataset** from a **COO CSV** file,
        transform time series data into supervised data,
        and split the dataset into several datasets.

        The default **float type** is ``float32``, you can change it to ``float64`` if needed.
        The default **device** is ``cpu``, you can change it to ``cuda`` or ``mps`` if needed.

        :param filename: type str COO CSV filename.
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param split_strategy: the strategy to split the dataset, can be 'intra' or 'inter', the default is 'intra'.
                                'intra' means splitting the dataset into several parts by the ratios,
                                'inter' means splitting the dataset by the time series.
        :param device: the device to load the data, default is 'cpu'.
                          This dataset device can be one of ['cpu', 'cuda', 'mps'].
                            This dataset device can be **different** to the model device.
        :param global_ts_ids: the global time series IDs, can be a list or a tensor.
                            If provided, the dataset will use these IDs instead of inferring from the COO data.
        :param global_time_point_ids: the global time point IDs, can be a list or a tensor.
                            If provided, the dataset will use these IDs instead of inferring from the COO data.
        :param global_variable_ids: the global variable IDs, can be a list or a tensor.
                            If provided, the dataset will use these IDs instead of inferring from the COO data

        :return: the (split) datasets as SMCDataset objects.
    """

    assert input_window_size > 0, f'Invalid input window size: {input_window_size}'
    assert output_window_size > 0, f'Invalid output window size: {output_window_size}'
    assert stride > 0, f'Invalid stride: {stride}'

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

        assert split_strategy in ('intra', 'inter'), \
            f"Invalid split strategy: {split_strategy}. The split strategy should be one of ['intra', 'inter']."

    float_type = np.float32  # the default float type is ``float32``, you can change it to ``float64`` if needed
    device = get_device(device)

    df = pd.read_csv(filename)

    coo_array = df.iloc[:, :-1].values.astype(np.int64)
    values_array = df.iloc[:, -1:].values.astype(float_type)
    coo_tensor = torch.tensor(coo_array, device=device)
    values_tensor = torch.tensor(values_array, device=device)

    smc_kwargs = {
        'coo': coo_tensor,
        'values': values_tensor,
        'input_window_size': input_window_size,
        'output_window_size': output_window_size,
        'horizon': horizon,
        'stride': stride,
        'global_ts_ids': global_ts_ids,
        'global_time_point_ids': global_time_point_ids,
        'global_variable_ids': global_variable_ids,
        'tqdm_disable': False,
    }

    if split_ratios is None:
        return SMCDataset(**smc_kwargs)

    smc_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])
    if split_strategy == 'intra':
        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            if i > 0:
                smc_kwargs.update({'stride': output_window_size})
            split_ds = SMCDataset(**smc_kwargs).split(s, e, is_strict=False, mark='intra_split_{}'.format(i))
            smc_datasets.append(split_ds)
    else:  # split_strategy == 'inter'
        ts_ids = torch.unique(coo_tensor[:, 0]) if global_ts_ids is None else global_ts_ids

        # shuffle `ts_ids` to avoid any potential bias
        if isinstance(ts_ids, list):
            ts_ids = torch.tensor(ts_ids, device=device)
        ts_ids = ts_ids[torch.randperm(len(ts_ids), device=ts_ids.device)]
        ts_num = len(ts_ids)

        for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
            start, end = int(ts_num * round(s, 10)), int(ts_num * round(e, 10))
            selected_ts_ids = ts_ids[start:end]

            coo_list = []
            values_list = []
            idx = torch.isin(coo_tensor[:, 0], selected_ts_ids)
            if torch.any(idx):
                coo_list.append(coo_tensor[idx])
                values_list.append(values_tensor[idx])

            if coo_list:
                split_coo = torch.cat(coo_list, dim=0)
                split_values = torch.cat(values_list, dim=0)
            else:
                split_coo = coo_tensor.new_empty((0, coo_tensor.size(1)), dtype=coo_tensor.dtype)
                split_values = values_tensor.new_empty((0, values_tensor.size(1)), dtype=values_tensor.dtype)

            if i > 0:
                smc_kwargs.update({'stride': output_window_size})
            split_kwargs = {**smc_kwargs, 'coo': split_coo, 'values': split_values, 'global_ts_ids': selected_ts_ids}
            split_smc_dataset = SMCDataset(**split_kwargs, mark='inter_split_{}'.format(i))
            split_smc_dataset.ratio = round(e - s, 10)
            smc_datasets.append(split_smc_dataset)

    return smc_datasets


def load_smir_datasets(filenames: List[Union[str, Tuple[str, str], List[str]]],
                       variables: List[str],
                       timepoint_variables: List[str],
                       ex_variables: List[str] = None,
                       mask_ex_variables: bool = False,
                       ex2_variables: List[str] = None,
                       supervised_strategy: AbstractSupervisedStrategy = None,
                       split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                       device: Union[Literal['cpu', 'cuda', 'mps'], str] = 'cpu',
                       transpose: bool = False,
                       show_loading_progress: bool = True,
                       max_loading_workers: int = None) -> SMIrDatasetSequence:
    """
        Load ** SMIrDataset ** from several **CSV** files, directories or a zip file,
        transform time series data into supervised data,
        and split the dataset into several datasets.

        :param filenames: list of CSV filenames, support CSV in a zip file.
        :param variables: names of the target variables, and can be one or more variables in the list.
        :param timepoint_variables: names of the time variables.
        :param ex_variables: names of the exogenous variables.
        :param mask_ex_variables: whether to mask the exogenous variables. This uses for fusing exogenous time series.
        :param ex2_variables: names of the second exogenous variables.
                            This is used for pre-known features, such as time features, forecasted weather factors, etc.
        :param supervised_strategy: the strategy to transform time series data into supervised data.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       This dataset device can be **different** to the model device.
        :param transpose: whether to transpose the time series data while reading (in zipped) CSV files.
        :param show_loading_progress: whether to show the loading progress bar.
        :param max_loading_workers: maximum number of threads for loading data, default is None (uses system default).
        :return: the (split) datasets as ``SMIrDataset`` objects.
    """

    if split_ratios is not None:
        if isinstance(split_ratios, (int, float)):
            split_ratios = [split_ratios]

        if not all(0 < ratio <= 1 for ratio in split_ratios) or sum(split_ratios) > 1:
            raise ValueError(f'Invalid split ratio: {split_ratios}. All ratios must be in (0, 1] and sum <= 1.')

    float_type = np.float32
    device = get_device(device)
    show_pregress = show_loading_progress
    max_workers = max_loading_workers

    if split_ratios is None:
        smir_args = get_smt_args_parallel(filenames, variables, True, ex_variables, mask_ex_variables,
                                          ex2_variables, timepoint_variables=timepoint_variables,
                                          float_type=float_type, device=device, transpose=transpose,
                                          show_progress=show_pregress, max_workers=max_workers)
        smir_args.update({'strategy': supervised_strategy, 'show_progress': show_pregress})
        return SMIrDataset(**smir_args)

    smir_datasets = []
    cum_split_ratios = np.cumsum([0, *split_ratios])

    filename_num = len(filenames)
    if filename_num < 2:
        raise ValueError(f'Invalid number of files: {filename_num}. '
                         f'The number of files should be at least 2 for "inter" split strategy.')

    for i, (s, e) in enumerate(zip(cum_split_ratios[:-1], cum_split_ratios[1:])):
        start, end = int(filename_num * round(s, 10)), int(filename_num * round(e, 10))
        split_filenames = filenames[start:end]
        smir_args = get_smt_args_parallel(split_filenames, variables, True, ex_variables, mask_ex_variables,
                                          ex2_variables, timepoint_variables=timepoint_variables,
                                          float_type=float_type, device=device, transpose=transpose,
                                          show_progress=show_pregress, max_workers=max_workers)
        smir_args.update({'strategy': supervised_strategy, 'show_progress': show_pregress})
        split_dataset = SMIrDataset(**smir_args, mark='inter_split_{}'.format(i))
        split_dataset.ratio = round(e - s, 10)
        smir_datasets.append(split_dataset)

    return smir_datasets
