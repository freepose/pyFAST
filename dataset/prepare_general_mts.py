#!/usr/bin/env python
# encoding: utf-8

"""

    The general time series datasets management module. Each time series dataset is loaded from a CSV file.

    The datasets are available at: https://zenodo.org/records/15255776

"""

import os
from typing import Literal, Tuple, Union

from fast.data import SSTDataset
from experiment.load import load_sst_dataset

metadata = {
    "ETTh1": {
        "path": "{root}/Github_ETT_small/ETTh1.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        }
    },
    "ETTh2": {
        "path": "{root}/Github_ETT_small/ETTh2.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        }
    },
    "ETTm1": {
        "path": "{root}/Github_ETT_small/ETTm1.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        }
    },
    "ETTm2": {
        "path": "{root}/Github_ETT_small/ETTm2.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
        }
    },
    "ExchangeRate": {
        "path": "{root}/Github_exchange_rate/exchange_rate.csv",
        "columns": {
            "time": "Date",
            "univariate": ["Singapore"],
            "multivariate": ["Australia", "British", "Canada", "Switzerland", "China", "Japan", "New Zealand",
                             "Singapore"]
        }
    },
    "JenaClimate": {
        "path": "{root}/MaxPlanck_Jena_Climate/mpi_roof_2010a.csv",
        "columns": {
            "time": "Date",
            "univariate": ["CO2 (ppm)"],
            "multivariate": ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
                             "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
                             "wv (m/s)",
                             "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m)", "PAR (mol/m/s)",
                             "max. PAR (mol/m/s)", "Tlog (degC)", "CO2 (ppm)"]
        }
    },
    "Electricity": {
        "path": "{root}/UCI_Electricity/electricity_20160701_20190702.csv",
        "columns": {
            "time": "Date",
            "univariate": ["320"],
            "multivariate": ["0", "320"]
        }
    },
    "PeMS03": {
        "path": "{root}/US_PEMS_03_04_07_08/pems03_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["357"],
            "multivariate": ["0", "357"]
        }
    },
    "PeMS04": {
        "path": "{root}/US_PEMS_03_04_07_08/pems04_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["306"],
            "multivariate": ["0", "306"]
        }
    },
    "PeMS07": {
        "path": "{root}/US_PEMS_03_04_07_08/pems07_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["882"],
            "multivariate": ["0", "882"]
        }
    },
    "PeMS08": {
        "path": "{root}/US_PEMS_03_04_07_08/pems08_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["169"],
            "multivariate": ["0", "169"]
        }
    },
    "Traffic": {
        "path": "{root}/US_PEMS_Traffic/traffic_20160701_20180702.csv",
        "columns": {
            "time": "Date",
            "univariate": ["861"],
            "multivariate": ["0", "861"]
        }
    },
    "US_CDC_Flu": {
        "path": "{root}/US_CDC_Flu_Activation_Level/US_regional_flu_level_20181004_20250322.csv",
        "columns": {
            "time": "Date",
            "univariate": ["Commonwealth of the Northern Mariana Islands"],
            "multivariate": ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
                             "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
                             "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
                             "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
                             "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
                             "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
                             "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
                             "West Virginia", "Wisconsin", "Wyoming", "New York City", "Puerto Rico", "Virgin Islands",
                             "Commonwealth of the Northern Mariana Islands"]
        }
    },
    "US_CDC_ILI": {
        "path": "{root}/US_CDC_ILI/US_National_ILI_1997_2025.csv",
        "columns": {
            "time": "Date",
            "univariate": ["ILITOTAL"],
            "multivariate": ["WEIGHTED ILI", "UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24",
                             "AGE 65", "ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"]
        }
    },
    "WHO_JAPAN_ILI": {
        "path": "{root}/WHO_Japan_ILI/Japan_ILI_19961006_20250309.csv",
        "columns": {
            "time": "Date",
            "univariate": ["ILI_ACTIVITY"],
            "multivariate": ["AH1", "AH1N12009", "AH3", "AH5", "ANOTSUBTYPED", "INF_A", "BVIC", "BYAM",
                             "BNOTDETERMINED", "INF_B", "INF_ALL", "INF_NEGATIVE", "ILI_ACTIVITY"]
        }
    }
}


def load_general_mts_sst(mts_data_root: str,
                         dataset_name: str,
                         task: Literal['univariate', 'multivariate'] = 'univariate',
                         is_time_feature: bool = False,
                         input_window_size: int = 96,
                         output_window_size: int = 24,
                         horizon: int = 1,
                         stride: int = 1,
                         train_ratio: float = 0.8,
                         val_ratio: float = None) -> Union[SSTDataset, Tuple[SSTDataset, ...]]:
    """
        Load general time series dataset from a CSV file, transform time series data into supervised data,
        and split the dataset into training, validation, and test sets.

        The datasets are available at: https://zenodo.org/records/15255776

        The default **float type** is ``float32``, you can change it  to ``float64`` if needed in ``load_sst_dataset()``.
        The default **device** is ``cpu``, you can change it to ``cuda`` if needed in ``load_sst_dataset()``.

        Example:
            >>> mts_data_root = os.path.join(os.path.expanduser('~/data'), 'time_series/general_mts')
            >>> ds = load_general_mts_sst(mts_data_root, 'ETTh1', 'univariate', train_ratio=0.7, val_ratio=0.1)

        :param mts_data_root: the general time series dataset root path.
        :param dataset_name: the name of the dataset.
        :param task: the task type, 'univariate' or 'multivariate'.
        :param is_time_feature: whether to use time features. Default is False.
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param train_ratio: the ratio of training set. Default is 0.8.
        :param val_ratio: the ratio of validation set. Default is None.
        :return: (train_ds, val_ds, test_ds): the datasets split into training, validation, and testing sets.
    """
    assert dataset_name in metadata, \
        f"Dataset '{dataset_name}' not found in metadata. The dataset name should be one of {list(metadata.keys())}."
    assert task in ['univariate', 'multivariate'], "Task should be 'univariate' or 'multivariate'."

    etth1_metadata = metadata[dataset_name]
    filename = etth1_metadata['path'].format(root=mts_data_root)
    variables = etth1_metadata['columns'][task]

    time_variable = etth1_metadata['columns']['time'] if is_time_feature else None
    frequency = 'D'
    is_time_normalized = True

    print('Loading {}'.format(filename))
    sst_params = {
        'filename': filename,
        'variables': variables, 'mask_variables': False,
        'ex_variables': None, 'mask_ex_variables': False,
        'time_variable': time_variable, 'frequency': frequency, 'is_time_normalized': is_time_normalized,
        'input_window_size': input_window_size, 'output_window_size': output_window_size,
        'horizon': horizon, 'stride': stride,
        'train_ratio': train_ratio, 'val_ratio': val_ratio
    }
    sst_datasets = load_sst_dataset(**sst_params)

    return sst_datasets
