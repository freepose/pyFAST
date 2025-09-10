#!/usr/bin/env python
# encoding: utf-8

"""

    The multisource time series dataset loading module. Each source is loaded from a CSV file.

    A dataset may consist of multiple sources, i.e, CSV files.
    All CSV files in a common datasets share the same data fields.

"""
import os, logging

from typing import Literal, Tuple, List, Union, Dict, Any
from pathlib import Path

from fast.data import SMTDataset, SMDDataset
from fast.data.processing.load import load_smx_datasets

smt_metadata = {
    # [Climate] Climate time series datasets are available from:
    # https://github.com/edebrouwer/gru_ode_bayes/blob/master/gru_ode_bayes/datasets/Climate/small_chunked_sporadic.csv
    # https://www.osti.gov/biblio/1394920

    "USHCN": {
        "paths": ["{root}/climate/US_Historical_Climatology_Network/02_multi_source_sparse"],
        "columns": {
            "names": ['ID', 'Time', 'Snow', 'SnowDepth', 'Precipitation', 'T_max', 'T_min'],
            "time": "Time",
            "univariate": ["T_max", "T_min"],
            "multivariate": ["Snow", "SnowDepth", "Precipitation", "T_max", "T_min"],
            "exogenous2": ["ID"]
        }
    },

    # [Disease] Glucose time series datasets, which are usually unaligned.
    "SH_diabetes": {
        "paths": ["{root}/disease/sh_diabetes/02_multi_source/T1DM",
                  "{root}/disease/sh_diabetes/02_multi_source/T2DM"],
        "columns": {
            "names": [
                'Date', 'CGM', 'CGB', 'Blood ketone', 'Dietary intake', 'Bolus insulin', 'Basal insulin',
                'Insulin dose s.c. id:0 medicine', 'Insulin dose s.c. id:0 dosage',
                'Insulin dose s.c. id:1 medicine', 'Insulin dose s.c. id:1 dosage',
                'Non-insulin id:0 medicine', 'Non-insulin id:0 dosage',
                'Non-insulin id:1 medicine', 'Non-insulin id:1 dosage',
                'Non-insulin id:2 medicine', 'Non-insulin id:2 dosage',
                'Insulin dose i.v. id:0 medicine', 'Insulin dose i.v. id:0 dosage',
                'Insulin dose i.v. id:1 medicine', 'Insulin dose i.v. id:1 dosage',
                'Insulin dose i.v. id:2 medicine', 'Insulin dose i.v. id:2 dosage'],
            "time": "Date",
            "univariate": ["CGM"],
            "multivariate": ["CGM"],  # univariate and multivariate are the same, change it if needed.
            "exogenous": ['Dietary intake', 'Bolus insulin', 'Basal insulin',
                          'Insulin dose s.c. id:0 dosage', 'Insulin dose s.c. id:1 dosage',
                          'Non-insulin id:0 dosage', 'Non-insulin id:1 dosage', 'Non-insulin id:2 dosage',
                          'Insulin dose i.v. id:0 dosage', 'Insulin dose i.v. id:1 dosage',
                          'Insulin dose i.v. id:2 dosage']  # these features are sparse_fusion
        }
    },

    "PhysioNet": {
        "paths": [
            "{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse/set-a",
            "{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse/set-b",
            "{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse/set-c"
        ],
        "columns": {
            "names": ['RecordID', 'Age', 'Gender', 'Height', 'Weight', 'ICUType', 'Albumin', 'ALP', 'ALT', 'AST',
                      'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3',
                      'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP',
                      'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI',
                      'TroponinT', 'Urine', 'WBC'],
            "univariate": ["Glucose"],
            "multivariate": ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine',
                             'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP',
                             # 'MechVent', # Mechanical ventilation respiration（0:false, 1:true）机械通气呼吸（0:否，1:是）
                             'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2', 'PaO2',
                             'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
                             'Urine', 'WBC'],
        }
    },

    # [Energy] Wind power time series datasets, some of them consist of exoogenous variables.
    "GreeceWPF": {
        "paths": ["{root}/energy_wind/GitHub_Greece_wind_energy_forecasting_2017_2020/02_multi_source/{freq}"],
        "freq": ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "univariate": ["Wind energy (MW)"],
            "multivariate": ["Wind energy (MW)"],  # univariate and multivariate are the same, change it if needed.
            "exogenous": ["airTemperature", "cloudCover", "gust", "humidity", "precipitation", "pressure",
                          "visibility", "windDirection", "windSpeed"]
        }
    },

    "GreeceWPF_1day": {
        "paths": ["{root}/energy_wind/GitHub_Greece_wind_energy_forecasting_2017_2020/02_multi_source/1day"],
        "columns": {
            "univariate": ["power(MW)"],
            "multivariate": ["power(MW)"],  # univariate and multivariate are the same, change it if needed.
            "exogenous": ["airTemperature", "cloudCover", "gust", "humidity", "precipitation", "pressure",
                          "visibility", "windDirection", "windSpeed"]
        }
    },

    "SDWPF": {
        "paths": ["{root}/energy_wind/KDDCup2022_Spatial_Dynamic_Wind_Power_Forecasting/02_multi_source_szw/{freq}"],
        "freq": ["1day"] + ["10min", "30min", "1hour", "6hour", "12hour", "1day"],
        "columns": {
            "univariate": ["Patv"],
            "multivariate": ["Patv"],
            "exogenous": ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv"]
        }
    },

    "SDWPF_1day": {
        "paths": ["{root}/energy_wind/KDDCup2022_Spatial_Dynamic_Wind_Power_Forecasting/02_multi_source_szw/1day"],
        "columns": {
            "univariate": ["Patv"],
            "multivariate": ["Patv"],
            "exogenous": ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv"]
        }
    },

    "WSTD2": {
        "paths": ["{root}/energy_wind/Zenodo_Wind_Spatio_Temporal_Dataset2_2010_2011/02_multi_source/{freq}"],
        "freq": ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "univariate": ["Power"],
            "multivariate": ["Power"],
        }
    },

    # [Spatio-temporal] Human activity recognition datasets, the "time" feature is normalized.
    # The original data is from: https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity
    "HumanActivity": {
        # "paths": ["{root}/../spatio_temporal/human_activity/02_multi_source/"],
        "paths": ["{root}/../spatio_temporal/human_activity/02_multi_source_sparse/"],
        "columns": {
            "univariate": ["010-000-024-033_x", "010-000-024-033_y", "010-000-024-033_z"],
            "multivariate": ["010-000-024-033_x", "010-000-024-033_y", "010-000-024-033_z",
                             "010-000-030-096_x", "010-000-030-096_y", "010-000-030-096_z",
                             "020-000-032-221_x", "020-000-032-221_y", "020-000-032-221_z",
                             "020-000-033-111_x", "020-000-033-111_y", "020-000-033-111_z"
                             ],
            "exogenous2": ["normalized_time"],
        }
    },

    # [Protein] pKa prediction datasets
    "phmd_2d_549_train": {
        "paths": ["{root}/protein_pKa/phmd_2d_549_sparse/train/"],
        "columns": {
            "names": [
                "PDB ID", "chain", "amino acid", "Res Name", "Res ID", "Titration", "Target_pKa", "model_pKa",
                "pKa shift", "res_name", *[str(i) for i in range(480)]
            ],
            "univariate": ["pKa_shift"],  # mask variable is 'Res Name'
            "multivariate": ["pKa_shift"],
            "exogenous": [str(i) for i in range(480)]
        }
    },
    "phmd_2d_549_val": {
        "paths": ["{root}/protein_pKa/phmd_2d_549_sparse/valid/"],
        "columns": {
            "names": [
                "PDB ID", "chain", "amino acid", "Res Name", "Res ID", "Titration", "Target_pKa", "model_pKa",
                "pKa shift", "res_name", *[str(i) for i in range(480)]
            ],
            "univariate": ["pKa_shift"],  # mask variable is 'Res Name'
            "multivariate": ["pKa_shift"],
            "exogenous": [str(i) for i in range(480)]
        }
    },
    "phmd_2d_549_test": {
        "paths": ["{root}/protein_pKa/phmd_2d_549_sparse/test/"],
        "columns": {
            "names": [
                "PDB ID", "chain", "amino acid", "Res Name", "Res ID", "Titration", "Target_pKa", "model_pKa",
                "pKa shift", "res_name", *[str(i) for i in range(480)]
            ],
            "univariate": ["pKa_shift"],  # mask variable is 'Res Name'
            "multivariate": ["pKa_shift"],
            "exogenous": [str(i) for i in range(480)]
        }
    },

    "GFM": {
        "paths": [
            "{root}/simulation/gfm_sim_abc/",
            # "{root}/simulation/gfm_sim_abc_simple/",
        ],
        "columns": {
            "names": ["simTime", "Vgq", "Vgd", "Pinertia", "Pdamping", "SCR", "XbyR", "IsOvercurrent",
                      "GridMag", "GridFreq", "GridPhase", "Pref_real", "Qref_real", "VolRef", "Qload",
                      "Igd", "Igq", "Ia", "Ib", "Ic", "Va", "Vb", "Vc"],
            "univariate": ["Vgd", "Vgq", "Igd", "Igq"],
            "multivariate": ["Vgd", "Vgq", "Igd", "Igq", "Pinertia", "Pdamping", "IsOvercurrent"],
            "exogenous": ["SCR", "XbyR", "GridMag", "GridFreq", "GridPhase",
                          "Pref_real", "Qref_real", "VolRef", "Qload"],
            "exogenous2": ["simTime"],
        }
    }
}


def prepare_smx_datasets(data_root: str,
                         dataset_name: str,
                         input_window_size: int = 96,
                         output_window_size: int = 24,
                         horizon: int = 1,
                         stride: int = 1,
                         split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                         split_strategy: Literal['intra', 'inter'] = 'intra',
                         device: Union[Literal['cpu', 'mps', 'cuda'], str] = 'cpu',
                         **task_kwargs: Dict[str, Any]) -> Union[SMTDataset, List[SMTDataset]]:
    """
        Prepare several SMTDataset/SMDDataset for machine/incremental learning tasks.

        The default **float type** is ``float32``, you can change it in ``load_sst_datasets()`` to ``float64`` if needed .

        :param data_root: the ``time_series`` directory.
        :param dataset_name: the name of the dataset.
        :param input_window_size: the input window size, i.e., the number of time steps in the input sequence.
        :param output_window_size: the output window size, i.e., the number of time steps in the output sequence.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
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
                            ``ts``: str, the time series type, 'univariate' or 'multivariate'.
                            ``ts_mask``: bool, whether to mask the time series variables, default is False.
                            ``use_ex``: bool, whether to use exogenous variables, default is None.
                            ``ex_ts_mask``: bool, whether to mask the exogenous variables, default is False.
                            ``use_ex2``: bool, whether to use time features, default is False.
                            ``dynamic_padding``: bool, whether to use dynamic padding for the time series, default is False.
        :return: the (split) datasets as SSTDataset objects.
    """
    assert dataset_name in smt_metadata, \
        f"Dataset '{dataset_name}' not found in metadata. The dataset name should be one of {list(smt_metadata.keys())}."
    given_metadata = smt_metadata[dataset_name]

    freq = given_metadata['freq'][0] if 'freq' in given_metadata else None
    paths = given_metadata['paths']
    paths = [os.path.normpath(path.format(root=data_root, freq=freq or '')) for path in paths]

    task_ts = task_kwargs.get('ts', 'univariate')
    task_ts_mask = task_kwargs.get('ts_mask', False)
    task_use_ex = task_kwargs.get('use_ex', False)
    task_ex_mask = task_kwargs.get('ex', False)
    task_ex2 = task_kwargs.get('use_ex2', False)
    task_dynamic_padding = task_kwargs.get('dynamic_padding', False)

    variables = given_metadata['columns'].get(task_ts, None)
    if variables is None:
        raise ValueError(f"Task type '{task_ts}' not found in dataset '{dataset_name}' metadata.")

    ex_variables = given_metadata['columns'].get('exogenous', None) if task_use_ex else None
    ex2_variables = given_metadata['columns'].get('exogenous2', None) if task_ex2 else None

    filenames = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Path not found: {path}')

        if path.is_file():
            filenames.append(path)
        elif path.is_dir():
            csv_files = list(path.glob('*.csv'))
            filenames.extend(csv_files)

    if len(filenames) == 0:
        raise FileNotFoundError(f'No CSV files found in paths: {paths}')

    filenames = sorted(filenames)
    logging.getLogger().info('Loading {} files in {}'.format(len(filenames), paths))
    # print('\n'.join([str(f) for f in filenames]))

    load_smt_args = {
        'filenames': filenames,
        'variables': variables, 'mask_variables': task_ts_mask,
        'ex_variables': ex_variables, 'mask_ex_variables': task_ex_mask,
        'ex2_variables': ex2_variables,
        'input_window_size': input_window_size, 'output_window_size': output_window_size,
        'horizon': horizon, 'stride': stride,
        'split_ratios': split_ratios,
        'split_strategy': split_strategy,
        'device': device,
        'ds_cls': SMDDataset if task_dynamic_padding else SMTDataset,
        'show_loading_progress': True,
        'max_loading_workers': None,
    }

    smt_datasets = load_smx_datasets(**load_smt_args)
    return smt_datasets


def verify_smt_datasets():
    """
        Verify the metadata.
    """
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'

    ds_names = list(smt_metadata.keys())
    for i, name in enumerate(ds_names):
        # task_config = {'ts': 'multivariate', 'ts_mask': True, 'use_ex': True0 , 'ex': True, 'use_ex2': True}
        # task_config = {'ts': 'multivariate'}
        task_config = {'ts': 'multivariate', 'ts_mask': True, 'use_ex': True, 'ex': True,
                       'dynamic_padding': False}
        print(i, end='\t')
        smt_datasets = prepare_smx_datasets(data_root, name, 10, 2, 1, 1, **task_config)
        smt_datasets = [smt_datasets] if isinstance(smt_datasets, (SMTDataset, SMDDataset)) else smt_datasets
        print('\n'.join([str(ds) for ds in smt_datasets]))
