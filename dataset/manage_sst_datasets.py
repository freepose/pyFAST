#!/usr/bin/env python
# encoding: utf-8

"""

    The single source time series datasets loading module. The time series dataset is loaded from a CSV file.

"""
import os, logging

from typing import Literal, List, Tuple, Union, Dict, Any
from fast.data import SSTDataset
from fast.data.processing import load_sst_datasets

sst_metadata = {
    # [Disease] Xiamen Center for Disease Control and Prevention (XMCDC): infectious disease surveillance data.
    # This is a built-in dataset in the pyFAST library.
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
            # "time": "Date",   # time format is 'YYYYWW', this cause errors.
            "univariate": ["手足口病"],
            "multivariate": ['手足口病', '肝炎', '其他感染性腹泻'],
            "exogenous": slice('平均温度', 'D7'),  # meteorological variables
        }
    },

    # [Energy] Electronic power load dataset.
    "SuzhouIPL": {
        "path": "{root}/energy_electronic_power_load/Suzhou_industrial_park_2016_2021/" +
                "01_single_source/senzhen_wu/ecpl_1hour_data_li.csv",
        "time_feature_freq": "h",
        "columns": {
            "time": "time",
            "univariate": ["residential"],
            "multivariate": ["commercial", "office", "public", "residential"],
            "exogenous": ["temperature", "humidity"]
        }
    },
    "SuzhouIPL_Sparse": {
        "path": "{root}/energy_electronic_power_load/Suzhou_industrial_park_2016_2021/" +
                "01_single_source/Suzhou_ipl_sparse_{freq}.csv",
        "freq": ["5min", "30min", "1hour"],
        "time_feature_freq": "5min",
        "columns": {
            "time": "Date",
            "univariate": ["Power_Residential"],
            "multivariate": ["Power_Commercial", "Power_Office", "Power_Public", "Power_Residential"],
            "exogenous": ["Temperature (℃)", "Humidity (%RH)"]
        }
    },

    # [Energy] Wind turbine active power time series datasets, some of these datasets include exogenous variables.
    # Public available at: TODO
    "TurkeyWPF": {
        "path": "{root}/energy_wind/Kaggle_Turkey_Wind_Turbine_Scada_Dataset_2018/Turkey_wind_turbine_{freq}.csv",
        "freq": ["10min", "30min", "1hour", "6hour", "12hour", "1day"],
        "columns": {
            "time": "Date",
            "univariate": ["LV ActivePower (kW)"],
            "multivariate": ["LV ActivePower (kW)"],  # NOTE: only one variable
            "exogenous": ["Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)", "Wind Direction (°)"]
        }
    },
    "TurkeyWPF_1day": {
        "path": "{root}/energy_wind/Kaggle_Turkey_Wind_Turbine_Scada_Dataset_2018/Turkey_wind_turbine_1day.csv",
        "columns": {
            "time": "Date",
            "univariate": ["LV ActivePower (kW)"],
            "multivariate": ["LV ActivePower (kW)"],  # NOTE: only one variable
            "exogenous": ["Wind Speed (m/s)", "Theoretical_Power_Curve (KWh)", "Wind Direction (°)"]
        }
    },

    "GreeceWPF": {
        "path": "{root}/energy_wind/GitHub_Greece_wind_energy_forecasting_2017_2020/01_single_source/Greece_power_{freq}.csv",
        "freq": ["1day"] + ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "time": "Date",
            "univariate": ["#36876_power(MW)"],
            "multivariate": slice("#32947_power(MW)", "#36876_power(MW)"),
        }
    },
    "GreeceWPF_1day": {
        "path": "{root}/energy_wind/GitHub_Greece_wind_energy_forecasting_2017_2020/01_single_source/Greece_power_1day.csv",
        "columns": {
            "time": "Date",
            "univariate": ["#36876_power(MW)"],
            "multivariate": slice("#32947_power(MW)", "#36876_power(MW)"),
            "exogenous": slice("#32947_airTemperature", "#36876_windSpeed")
        }
    },

    "SDWPF": {
        "path": "{root}/energy_wind/KDDCup2022_Spatial_Dynamic_Wind_Power_Forecasting/01_single_source/Patv_cubicspline_{freq}.csv",
        "freq": ["10min", "30min", "1hour", "6hour", "12hour", "1day"],
        "time_feature_freq": "10min",
        "columns": {
            "time": "Date",
            "univariate": ["134"],
            "multivariate": slice("1", "134"),
        }
    },

    "SDWPF_Sparse": {
        "path": "{root}/energy_wind/KDDCup2022_Spatial_Dynamic_Wind_Power_Forecasting/01_single_source_sparse/10min/[sparse_fusion]KDD22_wind_turbine_{freq}.csv",
        "freq": ["10min"],
        "time_feature_freq": "10min",
        "columns": {
            "time": "Date",
            "univariate": ["134_Patv"],
            "multivariate": slice("1_Patv", "134_Patv"),
        }
    },

    "WSTD2": {
        "path": "{root}/energy_wind/Zenodo_Wind_Spatio_Temporal_Dataset2_2010_2011/01_single_source/WSTD_linear_{freq}.csv",
        "freq": ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "time": "Date",
            "univariate": ["Turbine1_Power"],
            "multivariate": [f"Turbine{i}_Power" for i in range(1, 201)],
        }
    },
    "WSTD2_Sparse": {
        "path": "{root}/energy_wind/Zenodo_Wind_Spatio_Temporal_Dataset2_2010_2011/01_single_source/WSTD_sparse_{freq}.csv",
        "freq": ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "time": "Date",
            "univariate": ["Turbine1_Power"],
            "multivariate": [f"Turbine{i}_Power" for i in range(1, 201)],
        }
    },

    # General time series datasets are available at: https://zenodo.org/records/15255776
    "ETTh1": {
        "path": "{root}/general_mts/Github_ETT_small/ETTh1.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
        }
    },
    "ETTh2": {
        "path": "{root}/general_mts/Github_ETT_small/ETTh2.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
        }
    },
    "ETTm1": {
        "path": "{root}/general_mts/Github_ETT_small/ETTm1.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
        }
    },
    "ETTm2": {
        "path": "{root}/general_mts/Github_ETT_small/ETTm2.csv",
        "columns": {
            "time": "Date",
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
        }
    },
    "ExchangeRate_x1000": {
        "path": "{root}/general_mts/Github_exchange_rate/exchange_rate_x1000.csv",
        "columns": {
            "time": "Date",
            "univariate": ["Singapore"],
            "multivariate": ["Australia", "British", "Canada", "Switzerland", "China",
                             "Japan", "New Zealand", "Singapore"]
        }
    },
    "JenaClimate": {
        "path": "{root}/general_mts/MaxPlanck_Jena_Climate/mpi_roof_2010a.csv",
        "columns": {
            "time": "Date",
            "univariate": ["CO2 (ppm)"],
            "multivariate": ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
                             "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
                             "wv (m/s)", "max. wv (m/s)", "wd (deg)", "rain (mm)", "raining (s)", "SWDR (W/m)",
                             "PAR (mol/m/s)", "max. PAR (mol/m/s)", "Tlog (degC)", "CO2 (ppm)"]
        }
    },
    "Electricity": {
        "path": "{root}/general_mts/UCI_Electricity/electricity_20160701_20190702.csv",
        "columns": {
            "time": "Date",
            "univariate": ["320"],
            "multivariate": slice("0", "320")
        }
    },
    "PeMS03": {
        "path": "{root}/general_mts/US_PEMS_03_04_07_08/pems03_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["357"],
            "multivariate": slice("0", "357")
        }
    },
    "PeMS04": {
        "path": "{root}/general_mts/US_PEMS_03_04_07_08/pems04_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["306"],
            "multivariate": slice("0", "306")
        }
    },
    "PeMS07": {
        "path": "{root}/general_mts/US_PEMS_03_04_07_08/pems07_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["882"],
            "multivariate": slice("0", "882")
        }
    },
    "PeMS08": {
        "path": "{root}/general_mts/US_PEMS_03_04_07_08/pems08_flow.csv",
        "columns": {
            "time": "Date",
            "univariate": ["169"],
            "multivariate": slice("0", "169")
        }
    },
    "Traffic": {
        "path": "{root}/general_mts/US_PEMS_Traffic/traffic_20160701_20180702.csv",
        "columns": {
            "time": "Date",
            "univariate": ["861"],
            "multivariate": slice("0", "169")
        }
    },
    "US_CDC_Flu": {
        "path": "{root}/general_mts/US_CDC_Flu_Activation_Level/US_regional_flu_level_20181004_20250322.csv",
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
        "path": "{root}/general_mts/US_CDC_ILI/US_National_ILI_1997_2025.csv",
        "columns": {
            "time": "Date",
            "univariate": ["ILITOTAL"],
            "multivariate": ["WEIGHTED ILI", "UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24",
                             "AGE 65", "ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"]
        }
    },
    "WHO_JAPAN_ILI": {
        "path": "{root}/general_mts/WHO_Japan_ILI/Japan_ILI_19961006_20250309.csv",
        "columns": {
            "time": "Date",
            "univariate": ["ILI_ACTIVITY"],
            "multivariate": ["AH1", "AH1N12009", "AH3", "AH5", "ANOTSUBTYPED", "INF_A", "BVIC", "BYAM",
                             "BNOTDETERMINED", "INF_B", "INF_ALL", "INF_NEGATIVE", "ILI_ACTIVITY"]
        }
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
                         **task_kwargs: Dict[str, Any]) -> Union[SSTDataset, List[SSTDataset]]:
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

    freq = given_metadata['freq'][0] if 'freq' in given_metadata else None
    filename = given_metadata['path'].format(root=data_root, freq=freq or '')

    task_ts = task_kwargs.get('ts', 'univariate')
    task_ts_mask = task_kwargs.get('ts_mask', False)
    task_use_ex = task_kwargs.get('use_ex', False)
    task_ex_ts_mask = task_kwargs.get('ex_ts_mask', False)
    task_use_ex2 = task_kwargs.get('ex2', False)

    variables = given_metadata['columns'].get(task_ts, None)
    if variables is None:
        raise ValueError(f"Task type '{task_ts}' not found in dataset '{dataset_name}' metadata.")
    ex_variables = given_metadata['columns'].get('exogenous', None) if task_use_ex else None
    ex2_variables = given_metadata['columns'].get('exogenous2', None) if task_use_ex2 else None

    logging.getLogger().info('Loading {}'.format(filename))
    load_sst_args = {
        'filename': filename,
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
        Verify the metadata.
    """
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'

    ds_names = list(sst_metadata.keys())
    for i, name in enumerate(ds_names):
        task_config = {'ts': 'multivariate', 'ts_mask': True, 'use_ex': True, 'ex': True, 'use_ex2': True}
        # task_config = {'ts': 'multivariate'}
        print(i, end='\t')
        sst_datasets = prepare_sst_datasets(data_root, name, 48, 24, 1, 1, (0.7, 0.1, 0.2), **task_config)
        sst_datasets = [sst_datasets] if isinstance(sst_datasets, SSTDataset) else sst_datasets
        print('\n'.join([str(ds) for ds in sst_datasets]))
