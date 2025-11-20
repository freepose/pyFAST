#!/usr/bin/env python
# encoding: utf-8

"""

    Multi-source coordinate datasets for incomplete (sparse_fusion) time series forecasting (ITSF).

"""

import os, logging
import pandas as pd
import numpy as np
import torch

from typing import Literal, Tuple, List, Union, Dict, Any
from pathlib import Path

from fast import get_device
from fast.data import SMCDataset
from fast.data.processing import load_smc_datasets, SMCDatasetSequence

coo_metadata = {

    "PhysioNet": {
        "path": "{root}/disease/PhysioNet_Challenge_2012/" +
                "02_multi_source_coo/[coo+indexed]physionet_indexed_records.csv",
        "columns": {
            "names": ["id_index", "time_index", "parameter_index", "value"],
            "ts_ids": [i for i in range(11988)],
            "time_point_ids": [i for i in range(1440 * 2)],
            "variable_ids": [i for i in range(37)],
            # patient ids too many to list here
            "time_points": ([f'{hour:02d}:{minute:02d}' for hour in range(48) for minute in range(60)] + ['48:00'])[1:],
            "parameters": ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine',
                           'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent',
                           'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate',
                           'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC', 'Weight', 'pH'],
        }
    },

    "MIMIC-III_v1.4": {
        "path": "{root}/disease/PhysioNet_MIMIC-III_v1.4/02_multi_source_coo/[coo]identified_records_1min.csv",
        "columns": {
            "names": ["ID_index", "Date_index", "Variable_index", "Value"],
            "ts_ids": [i for i in range(23457)],
            "time_point_ids": [i for i in range(1440 * 2)],
            "variable_ids": [i for i in range(96)],
            # patient ids too many to list here
            "time_points": ([f'{hour:02d}:{minute:02d}' for hour in range(48) for minute in range(60)] + ['48:00'])[1:],
            "parameters": ['ID', 'Date', 'Alanine Aminotransferase (ALT)', 'Albumin', 'Albumin 5%',
                           'Alkaline Phosphatase', 'Anion Gap', 'Asparate Aminotransferase (AST)', 'Aspirin Drug',
                           'Base Excess', 'Basophils', 'Bicarbonate', 'Bilirubin, Total', 'Bisacodyl Drug',
                           'Calcium Gluconate', 'Calcium, Total', 'Calculated Total CO2', 'Chest Tube #1',
                           'Chest Tube #2', 'Chloride', 'Condom Cath', 'Creatinine', 'D5 1/2NS', 'D5W Drug',
                           'Dextrose 5%', 'Docusate Sodium Drug', 'Eosinophils', 'Fecal Bag', 'Foley',
                           'Furosemide (Lasix)', 'GT Flush', 'Gastric Gastric Tube', 'Gastric Meds', 'Glucose',
                           'Hematocrit', 'Hemoglobin', 'Heparin Sodium', 'Humulin-R Insulin Drug', 'Hydralazine',
                           'Insulin - Glargine', 'Insulin - Humalog', 'Insulin - Regular', 'Jackson Pratt #1', 'K Phos',
                           'KCL (Bolus)', 'LR', 'Lactate', 'Lorazepam (Ativan)', 'Lymphocytes', 'MCH', 'MCHC', 'MCV',
                           'Magnesium', 'Magnesium Sulfate', 'Magnesium Sulfate (Bolus)', 'Magnesium Sulfate Drug',
                           'Metoprolol', 'Metoprolol Tartrate Drug', 'Midazolam (Versed)', 'Monocytes',
                           'Morphine Sulfate', 'Neutrophils', 'Nitroglycerin', 'Norepinephrine', 'OR Cell Saver Intake',
                           'OR Crystalloid Intake', 'OR EBL', 'Ostomy (output)', 'PO Intake', 'PT', 'PTT',
                           'Packed Red Blood Cells', 'Pantoprazole Drug', 'Phenylephrine', 'Phosphate', 'Piggyback',
                           'Platelet Count', 'Potassium', 'Potassium Chloride', 'Potassium Chloride Drug',
                           'Pre-Admission', 'RDW', 'Red Blood Cells', 'Sodium', 'Sodium Chloride 0.9%  Flush Drug',
                           'Solution', 'Specific Gravity', 'Sterile Water', 'Stool Out Stool', 'TF Residual',
                           'Ultrafiltrate Ultrafiltrate', 'Urea Nitrogen', 'Urine Out Incontinent', 'Void',
                           'White Blood Cells', 'pCO2', 'pH', 'pO2'],
        }
    },

    "MIMIC-III-Ext-tPatchGNN": {
        "path": "{root}/disease/PhysioNet_MIMIC-III-Ext-tPatchGNN/02_multi_source_coo/[coo]identified_records_1min.csv",
        "columns": {
            "names": ["ID_index", "Date_index", "Variable_index", "Value"],
            "ts_ids": [i for i in range(23457)],
            "time_point_ids": [i for i in range(1440 * 2)],
            "variable_ids": [i for i in range(96)],
            # patient ids too many to list here
            "time_points": ([f'{hour:02d}:{minute:02d}' for hour in range(48) for minute in range(60)] + ['48:00'])[1:],
            "parameters": ['ID', 'Date', 'Alanine Aminotransferase (ALT)', 'Albumin', 'Albumin 5%',
                           'Alkaline Phosphatase', 'Anion Gap', 'Asparate Aminotransferase (AST)', 'Aspirin Drug',
                           'Base Excess', 'Basophils', 'Bicarbonate', 'Bilirubin, Total', 'Bisacodyl Drug',
                           'Calcium Gluconate', 'Calcium, Total', 'Calculated Total CO2', 'Chest Tube #1',
                           'Chest Tube #2', 'Chloride', 'Condom Cath', 'Creatinine', 'D5 1/2NS', 'D5W Drug',
                           'Dextrose 5%', 'Docusate Sodium Drug', 'Eosinophils', 'Fecal Bag', 'Foley',
                           'Furosemide (Lasix)', 'GT Flush', 'Gastric Gastric Tube', 'Gastric Meds', 'Glucose',
                           'Hematocrit', 'Hemoglobin', 'Heparin Sodium', 'Humulin-R Insulin Drug', 'Hydralazine',
                           'Insulin - Glargine', 'Insulin - Humalog', 'Insulin - Regular', 'Jackson Pratt #1', 'K Phos',
                           'KCL (Bolus)', 'LR', 'Lactate', 'Lorazepam (Ativan)', 'Lymphocytes', 'MCH', 'MCHC', 'MCV',
                           'Magnesium', 'Magnesium Sulfate', 'Magnesium Sulfate (Bolus)', 'Magnesium Sulfate Drug',
                           'Metoprolol', 'Metoprolol Tartrate Drug', 'Midazolam (Versed)', 'Monocytes',
                           'Morphine Sulfate', 'Neutrophils', 'Nitroglycerin', 'Norepinephrine', 'OR Cell Saver Intake',
                           'OR Crystalloid Intake', 'OR EBL', 'Ostomy (output)', 'PO Intake', 'PT', 'PTT',
                           'Packed Red Blood Cells', 'Pantoprazole Drug', 'Phenylephrine', 'Phosphate', 'Piggyback',
                           'Platelet Count', 'Potassium', 'Potassium Chloride', 'Potassium Chloride Drug',
                           'Pre-Admission', 'RDW', 'Red Blood Cells', 'Sodium', 'Sodium Chloride 0.9%  Flush Drug',
                           'Solution', 'Specific Gravity', 'Sterile Water', 'Stool Out Stool', 'TF Residual',
                           'Ultrafiltrate Ultrafiltrate', 'Urea Nitrogen', 'Urine Out Incontinent', 'Void',
                           'White Blood Cells', 'pCO2', 'pH', 'pO2'],
        }
    },

    "MIMIC-IV_v3.1": {
        "path": "{root}/disease/PhysioNet_MIMIC-IV_v3.1/02_multi_source_coo/[coo]identified_records_1min.csv",
        "columns": {
            "names": ["ID_index", "Date_index", "Variable_index", "Value"],
            "ts_ids": [i for i in range(22018)],
            "time_point_ids": [i for i in range(1440 * 2)],
            "variable_ids": [i for i in range(96)],
            # patient ids too many to list here
            "time_points": ([f'{hour:02d}:{minute:02d}' for hour in range(48) for minute in range(60)] + ['48:00'])[1:],
            "parameters": ['ID', 'Date', 'Acetaminophen-IV', 'Alanine Aminotransferase (ALT)', 'Albumin', 'Albumin 5%',
                           'Alkaline Phosphatase', 'Anion Gap', 'Asparate Aminotransferase (AST)', 'Base Excess',
                           'Basophils', 'Bicarbonate', 'Bilirubin, Total', 'Calcium Gluconate', 'Calcium, Total',
                           'Calculated Total CO2', 'Cefazolin', 'Cefepime', 'Ceftriaxone', 'Chloride', 'Creatinine',
                           'Dexmedetomidine (Precedex)', 'Dextrose 5%', 'Emesis', 'Eosinophils', 'Famotidine (Pepcid)',
                           'Fecal Bag', 'Fentanyl', 'Fentanyl (Concentrate)', 'Foley', 'Free Water',
                           'Furosemide (Lasix)', 'GT Flush', 'Gastric Meds', 'Glucose', 'Hematocrit', 'Hemoglobin',
                           'Heparin Sodium', 'Heparin Sodium (Prophylaxis)', 'Hydralazine', 'Hydromorphone (Dilaudid)',
                           'Insulin - Glargine', 'Insulin - Humalog', 'Insulin - Regular', 'K Phos', 'KCL (Bolus)',
                           'Lactate', 'Lorazepam (Ativan)', 'Lymphocytes', 'MCH', 'MCV', 'Magnesium',
                           'Magnesium Sulfate', 'Magnesium Sulfate (Bolus)', 'Metoprolol', 'Metronidazole',
                           'Midazolam (Versed)', 'Monocytes', 'Morphine Sulfate', 'Nasogastric', 'Neutrophils',
                           'Nitroglycerin', 'Norepinephrine', 'OR Cell Saver Intake', 'OR EBL', 'OR Urine',
                           'Oral Gastric', 'PT', 'PTT', 'Packed Red Blood Cells', 'Pantoprazole (Protonix)',
                           'Pantoprazole (Protonix) Continuous', 'Phenylephrine', 'Phosphate', 'Piggyback',
                           'Piperacillin/Tazobactam (Zosyn)', 'Platelet Count', 'Potassium', 'Potassium Chloride',
                           'Pre-Admission', 'Propofol', 'RDW', 'Red Blood Cells', 'Sodium', 'Solution',
                           'Specific Gravity', 'Sterile Water', 'Stool', 'Straight Cath', 'TF Residual',
                           'TF Residual Output', 'Urea Nitrogen', 'Vancomycin', 'Void', 'White Blood Cells', 'pCO2',
                           'pH', 'pO2'],
        }
    },

}


def prepare_smc_datasets(data_root: str,
                         dataset_name: str,
                         input_window_size: int = 96,
                         output_window_size: int = 24,
                         horizon: int = 1,
                         stride: int = 1,
                         split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                         split_strategy: Literal['intra', 'inter'] = 'intra',
                         device: Union[Literal['cpu', 'mps', 'cuda'], str] = 'cpu') -> SMCDatasetSequence:
    assert dataset_name in coo_metadata, \
        f"Dataset '{dataset_name}' not found in metadata. The dataset name should be one of {list(coo_metadata.keys())}."

    given_metadata = coo_metadata[dataset_name]
    coo_filename = given_metadata['path'].format(root=data_root)

    ts_ids = given_metadata['columns'].get('ts_ids', None)
    time_point_ids = given_metadata['columns'].get('time_point_ids', None)
    variable_ids = given_metadata['columns'].get('variable_ids', None)

    logging.getLogger().info('Loading {}'.format(coo_filename))
    load_smc_args = {
        "filename": coo_filename,
        "input_window_size": input_window_size,
        "output_window_size": output_window_size,
        "horizon": horizon,
        "stride": stride,
        "split_ratios": split_ratios,
        "split_strategy": split_strategy,
        "device": device,
        "global_ts_ids": ts_ids,
        "global_time_point_ids": time_point_ids,
        "global_variable_ids": variable_ids,
    }
    smc_datasets = load_smc_datasets(**load_smc_args)

    return smc_datasets
