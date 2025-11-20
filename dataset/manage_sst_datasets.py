#!/usr/bin/env python
# encoding: utf-8

"""

    The single source time series datasets loading module. The time series dataset is loaded from a CSV file.

"""
import os, logging

from typing import Literal, List, Tuple, Union, Dict, Any

from fast.data import SSTDataset
from fast.data.processing import load_sst_datasets, SSTDatasetSequence

sst_metadata = {
    # [Disease] [Built-in] Xiamen Center for Disease Control and Prevention (XMCDC): infection surveillance data.
    "XMCDC_1day": {
        "path": "../../dataset/xmcdc/outpatients_2011_2020_1day.csv",
        "columns": {
            "names": ['Date', '平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                      '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压', '最高风速',
                      '平均风速', '最低风速', 'BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise',
                      'BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise',
                      'BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise',
                      '手足口病', '肝炎', '其他感染性腹泻'],
            "univariate": ["手足口病"],
            "multivariate": ['手足口病', '肝炎', '其他感染性腹泻'],
            "exogenous": slice('平均温度', '最低风速'),  # meteorological variables
        }
    },
    "XMCDC_1week": {
        "path": "../../dataset/xmcdc/outpatients_2011_2020_1week.csv",
        "columns": {
            "names": ['Date', '平均温度', '最高温', '最低温', '平均降水', '最高露点', '平均露点', '最低露点',
                      '最高湿度(%)', '最低湿度(%)', '平均相对湿度(%)', '最高气压', '平均气压', '最低气压', '最高风速',
                      '平均风速', '最低风速', 'BSI_厦门手足口病_all', 'BSI_厦门手足口病_pc', 'BSI_厦门手足口病_wise',
                      'BSI_厦门肝炎_all', 'BSI_厦门肝炎_pc', 'BSI_厦门肝炎_wise',
                      'BSI_厦门腹泻_all', 'BSI_厦门腹泻_pc', 'BSI_厦门腹泻_wise',
                      '手足口病', '肝炎', '其他感染性腹泻',
                      'HF1', 'HF2', 'HF3', 'HF4', 'HF5', 'HF6', 'HF7',
                      'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7',
                      'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'],
            "univariate": ["手足口病"],
            "multivariate": ['手足口病', '肝炎', '其他感染性腹泻'],
            "exogenous": slice('平均温度', '最低风速'),  # meteorological variables
        }
    },

    # [General_MTS] General time series datasets are available at: https://zenodo.org/records/15255776
    "ETTh1": {
        "path": ("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTh1.csv"),
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ETTh2": {
        "path": ("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTh2.csv"),
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ETTm1": {
        "path": ("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTm1.csv"),
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ETTm2": {
        "path": ("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTm2.csv"),
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"],
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ExchangeRate": {
        "path": ("{root}/general_mts.zip", "general_mts/Github_exchange_rate/exchange_rate.csv"),
        "columns": {
            "univariate": ["Singapore"],
            "multivariate": ["Australia", "British", "Canada", "Switzerland", "China", "Japan",
                             "New Zealand", "Singapore"],
            "exogenous2": ['day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "JenaClimate": {
        "path": ("{root}/general_mts.zip", "general_mts/MaxPlanck_Jena_Climate/mpi_roof_2010a.csv"),
        "columns": {
            "univariate": ["CO2 (ppm)"],
            "multivariate": ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
                             "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
                             "wv (m/s)", "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m)",
                             "PAR (mol/m/s)", "max. PAR (mol/m/s)", "Tlog (degC)", "CO2 (ppm)"],
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "Electricity": {
        "path": ("{root}/general_mts.zip", "general_mts/UCI_Electricity/electricity_20160701_20190702.csv"),
        "columns": {
            "univariate": ["320"],
            "multivariate": slice("0", "320"),
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS03": {
        "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems03_flow.csv"),
        "columns": {
            "univariate": ["357"],
            "multivariate": slice("0", "357"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS04": {
        "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems04_flow.csv"),
        "columns": {
            "univariate": ["306"],
            "multivariate": slice("0", "306"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS07": {
        "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems07_flow.csv"),
        "columns": {
            "univariate": ["882"],
            "multivariate": slice("0", "882"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS08": {
        "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems08_flow.csv"),
        # "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems08_speed.csv"),
        # "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems08_occupancy.csv"),
        "columns": {
            "univariate": ["169"],
            "multivariate": slice("0", "169"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS-bay": {
        "path": "{root}/general_mts/US_PeMS-bay_20170101_20170630/00_extract/" + \
                "[downsample+wide]PeMS-bay_20170101_20170630_5min.csv",  # downsample as the same frequency (5min)
        "columns": {
            "univariate": ["400001"],
            "multivariate": slice("400001", "414694"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "metr-la": {
        "path": ("{root}/general_mts.zip",
                 "general_mts/US_metr-la_201203_201206/00_extract/[wide]metr_la_speed_201203_201206_5min.csv"),
        "columns": {
            "univariate": ["773869"],
            "multivariate": slice("773869", "769373"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "Traffic": {
        "path": ("{root}/general_mts.zip", "general_mts/US_PEMS_Traffic/traffic_20160701_20180702.csv"),
        "columns": {
            "univariate": ["861"],
            "multivariate": slice("0", "169"),
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "US_CDC_Flu-sparse": {
        # This is a sparse dataset
        "path": ("{root}/general_mts.zip",
                 "general_mts/US_CDC_Flu_Activation_Level/US_regional_flu_level_20181004_20250322.csv"),
        "columns": {
            "univariate": ["Commonwealth of the Northern Mariana Islands"],
            "multivariate": ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
                             "Delaware", "District of Columbia", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
                             "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
                             "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
                             "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
                             "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
                             "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
                             "West Virginia", "Wisconsin", "Wyoming", "New York City", "Puerto Rico", "Virgin Islands",
                             "Commonwealth of the Northern Mariana Islands"],
            "exogenous2": ['day_of_month', 'week_of_year']
        }
    },
    "US_CDC_ILI": {
        "path": ("{root}/general_mts.zip", "general_mts/US_CDC_ILI/US_National_ILI_1997_2025.csv"),
        "columns": {
            "univariate": ["ILITOTAL"],
            "multivariate": ["WEIGHTED ILI", "UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24",
                             "AGE 65", "ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"],
            "exogenous2": ['day_of_month', 'week_of_year']
        }
    },
    "WHO_JAPAN_ILI": {
        "path": ("{root}/general_mts.zip", "general_mts/WHO_Japan_ILI/Japan_ILI_19961006_20250309.csv"),
        "columns": {
            "univariate": ["ILI_ACTIVITY"],
            "multivariate": ["AH1", "AH1N12009", "AH3", "AH5", "ANOTSUBTYPED", "INF_A", "BVIC", "BYAM",
                             "BNOTDETERMINED", "INF_B", "INF_ALL", "INF_NEGATIVE", "ILI_ACTIVITY"],
            "exogenous2": ['day_of_month', 'week_of_year']
        }
    },

    "dka_1day": {
        "path": "{root}/energy_solar/heywhale_desert_knowledge_australia_2013_2014/" + \
                "01_single_source/[downsample+sparse]91-Site_1A-Trina_5W_1day.csv",
        "columns": {
            "names": ['Date', 'Active Energy Delivered-Received (kWh)', 'Current Phase Average (A)',
                      'Active Power (kW)', 'Wind Speed (m/s)', 'Weather Temperature Celsius (°C)',
                      'Weather Relative Humidity (%)', 'Global Horizontal Radiation (W/m²)',
                      'Diffuse Horizontal Radiation (W/m²)', 'Wind Direction (Degrees)', 'Weather Daily Rainfall (mm)'],
            "univariate": ["Active Power (kW)"],
            "multivariate": ["Active Power (kW)"],
            "exogenous": slice('Wind Speed (m/s)', 'Weather Daily Rainfall (mm)'),
        },
    },

}


def prepare_sst_datasets(data_root: str,
                         dataset_name: str,
                         input_window_size: int = 96,
                         output_window_size: int = 24,
                         horizon: int = 1,
                         stride: int = 1,
                         split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                         device: Union[Literal['cpu', 'mps', 'cuda'], str] = 'cpu',
                         **task_kwargs: Dict[str, Any]) -> SSTDatasetSequence:
    """
        Prepare several SSTDataset for machine learning tasks.

        The default **float type** is ``float32``, you can change it in ``load_sst_datasets()`` to ``float64`` if needed .

        :param data_root: the ``time_series`` directory.
        :param dataset_name: the name of the dataset.
        :param input_window_size: input window size of the transformed supervised data. A.k.a., lookback window size.
        :param output_window_size: output window size of the transformed supervised data. A.k.a., prediction length.
        :param horizon: the distance between input and output windows of a sample.
        :param stride: the distance between two consecutive samples.
        :param split_ratios: the ratios of consecutive split datasets. For example,
                            (0.7, 0.1, 0.2) means 70% for training, 10% for validation, and 20% for testing.
                            The default is none, which means non-split.
        :param device: the device to load the data, default is 'cpu'.
                       This dataset device can be one of ['cpu', 'cuda', 'mps'].
                       the dataset device can be **different** to the model device.
        :param task_kwargs: task settings for the dataset.
                            ``ts``: str, the time series type, 'univariate' or 'multivariate'.
                            ``ts_mask``: bool, whether to mask the time series variables, default is False.
                            ``use_ex``: bool, whether to use exogenous variables, default is None.
                            ``ex_ts_mask``: bool, whether to mask the exogenous variables, default is False.
                            ``use_ex2``: bool, whether to use time features, default is False.
                            ``random_mask``: type AbstractMask, the random mask generator, default is None.

        :return: the (split) datasets as SSTDataset objects.
    """
    assert dataset_name in sst_metadata, \
        f"Dataset '{dataset_name}' not found in metadata. The dataset name should be one of {list(sst_metadata.keys())}."
    given_metadata = sst_metadata[dataset_name]

    path = given_metadata['path']
    if isinstance(path, str):
        path = os.path.normpath(path.format(root=data_root))
        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: {path}')
    elif (isinstance(path, tuple) or isinstance(path, list)) and len(path) == 2:
        path = (os.path.normpath(path[0].format(root=data_root)), path[1])
    else:
        raise ValueError(f'Invalid path: {path}. It should be a string or a tuple/list of two strings.')

    task_ts = task_kwargs.get('ts', 'univariate')
    task_ts_mask = task_kwargs.get('ts_mask', False)
    task_use_ex = task_kwargs.get('use_ex', False)
    task_ex_ts_mask = task_kwargs.get('ex_ts_mask', False)
    task_use_ex2 = task_kwargs.get('use_ex2', False)

    variables = given_metadata['columns'].get(task_ts, None)
    if variables is None:
        raise ValueError(f"Task type '{task_ts}' not found in dataset '{dataset_name}' metadata.")
    ex_variables = given_metadata['columns'].get('exogenous', None) if task_use_ex else None
    ex2_variables = given_metadata['columns'].get('exogenous2', None) if task_use_ex2 else None

    logging.getLogger().info('Loading {}'.format(path))
    load_sst_args = {
        'path': path,
        'variables': variables, 'mask_variables': task_ts_mask,
        'ex_variables': ex_variables, 'mask_ex_variables': task_ex_ts_mask,
        'ex2_variables': ex2_variables,
        'input_window_size': input_window_size, 'output_window_size': output_window_size,
        'horizon': horizon, 'stride': stride,
        'split_ratios': split_ratios,
        'device': device,
    }
    sst_datasets = load_sst_datasets(**load_sst_args)

    return sst_datasets


def verify_sst_datasets():
    """
        Verify the ``SSTDataset`` metadata.
    """
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'

    ds_names = list(sst_metadata.keys())[2:]
    for i, name in enumerate(ds_names):
        # task_config = {'ts': 'multivariate'}
        task_config = {'ts': 'multivariate', 'ts_mask': True, 'use_ex': True, 'ex_ts_mask': True, 'use_ex2': True}
        print(i, end='\t')
        sst_datasets = prepare_sst_datasets(data_root, name, 48, 24, 1, 1, (0.7, 0.1, 0.2), **task_config)
        sst_datasets = [sst_datasets] if isinstance(sst_datasets, SSTDataset) else sst_datasets
        print('\n'.join([str(ds) for ds in sst_datasets]))
