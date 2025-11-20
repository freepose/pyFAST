#!/usr/bin/env python
# encoding: utf-8

"""

    The multisource **irregular** time series dataset loading module. Each source is loaded from a CSV file.

    A dataset may consist of multiple sources, i.e, CSV files.
    All CSV files in a common datasets share the same data fields.

"""

import os, random, logging, zipfile

from typing import Literal, Tuple, List, Union, Dict, Any, Optional
from pathlib import Path

from fast.data.processing import load_smir_datasets, retrieve_files_in_zip, SMIrDatasetSequence
from fast.data.smir_dataset import AbstractSupervisedStrategy

multi_source_irregular_metadata: Dict[str, Any] = {

    # [Disease-ICU]
    "PhysioNet-ir": {
        "paths": [
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_irregular.zip",
             "02_multi_source_irregular/set-a"),
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_irregular.zip",
             "02_multi_source_irregular/set-b"),
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_irregular.zip",
             "02_multi_source_irregular/set-c"),
        ],
        "columns": {
            # some demographics are removed in the forecasting task: 'Age', 'Gender', 'Height', 'ICUType', 'Weight'
            "names": ["Time", "normalized_time", "deltas", "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
                      "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
                      "Lactate", "Mg", "MAP", "MechVent",
                      # Mechanical ventilation respiration（0:false, 1:true）机械通气呼吸（0:否，1:是）
                      "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
                      "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC"],
            "univariate": ["Glucose"],
            "multivariate": ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
                             "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP",
                             "MechVent",  # Mechanical ventilation respiration（0:false, 1:true）机械通气呼吸（0:否，1:是）
                             "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2",
                             "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT",
                             "Urine", "WBC"],
            "timepoint": ['normalized_time'],
            "exogenous": ["normalized_time"],
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-III-Ext-tPatchGNN-ir": {
        "paths": [
            # ("{root}/disease/PhysioNet_MIMIC-III-Ext-tPatchGNN/02_multi_source_irregular.zip",
            #  "02_multi_source_irregular/1min"),
            ("{root}/disease/PhysioNet_MIMIC-III-Ext-tPatchGNN/02_multi_source_irregular.zip",
             "02_multi_source_irregular/1hour"),
        ],
        "columns": {
            "names": ["Date", "normalized_time", "deltas",
                      "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%", "Alkaline Phosphatase", "Anion Gap",
                      "Asparate Aminotransferase (AST)", "Aspirin Drug", "Base Excess", "Basophils",
                      "Bicarbonate", "Bilirubin, Total", "Bisacodyl Drug", "Calcium Gluconate", "Calcium, Total",
                      "Calculated Total CO2", "Chest Tube #1", "Chest Tube #2", "Chloride", "Condom Cath", "Creatinine",
                      "D5 1/2NS", "D5W Drug", "Dextrose 5%", "Docusate Sodium Drug", "Eosinophils", "Fecal Bag",
                      "Foley", "Furosemide (Lasix)", "GT Flush", "Gastric Gastric Tube", "Gastric Meds", "Glucose",
                      "Hematocrit", "Hemoglobin", "Heparin Sodium", "Humulin-R Insulin Drug", "Hydralazine",
                      "Insulin - Glargine", "Insulin - Humalog", "Insulin - Regular", "Jackson Pratt #1", "K Phos",
                      "KCL (Bolus)", "LR", "Lactate", "Lorazepam (Ativan)", "Lymphocytes", "MCH", "MCHC", "MCV",
                      "Magnesium", "Magnesium Sulfate", "Magnesium Sulfate (Bolus)", "Magnesium Sulfate Drug",
                      "Metoprolol", "Metoprolol Tartrate Drug", "Midazolam (Versed)", "Monocytes", "Morphine Sulfate",
                      "Neutrophils", "Nitroglycerin", "Norepinephrine", "OR Cell Saver Intake", "OR Crystalloid Intake",
                      "OR EBL", "Ostomy (output)", "PO Intake", "PT", "PTT", "Packed Red Blood Cells",
                      "Pantoprazole Drug", "Phenylephrine", "Phosphate", "Piggyback", "Platelet Count", "Potassium",
                      "Potassium Chloride", "Potassium Chloride Drug", "Pre-Admission", "RDW", "Red Blood Cells",
                      "Sodium", "Sodium Chloride 0.9%  Flush Drug", "Solution", "Specific Gravity", "Sterile Water",
                      "Stool Out Stool", "TF Residual", "Ultrafiltrate Ultrafiltrate", "Urea Nitrogen",
                      "Urine Out Incontinent", "Void", "White Blood Cells", "pCO2", "pH", "pO2"],
            "univariate": ["Glucose"],
            "multivariate": slice("Alanine Aminotransferase (ALT)", "pO2"),
            "timepoint": ['normalized_time'],
            "exogenous": ["normalized_time"],
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-III-v1.4-ir": {
        "paths": [("{root}/disease/PhysioNet_MIMIC-III_v1.4/02_multi_source_irregular.zip",
                   "02_multi_source_irregular/1min")],
        "columns": {
            "names": ["Date", "normalized_time", "deltas",
                      "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%", "Alkaline Phosphatase", "Anion Gap",
                      "Asparate Aminotransferase (AST)", "Aspirin Drug", "Base Excess", "Basophils", "Bicarbonate",
                      "Bilirubin, Total", "Bisacodyl Drug", "Calcium Gluconate", "Calcium, Total",
                      "Calculated Total CO2", "Chest Tube #1", "Chest Tube #2", "Chloride", "Condom Cath", "Creatinine",
                      "D5 1/2NS", "D5W Drug", "Dextrose 5%", "Docusate Sodium Drug", "Eosinophils", "Fecal Bag",
                      "Foley", "Furosemide (Lasix)", "GT Flush", "Gastric Gastric Tube", "Gastric Meds", "Glucose",
                      "Hematocrit", "Hemoglobin", "Heparin Sodium", "Humulin-R Insulin Drug", "Hydralazine",
                      "Insulin - Glargine", "Insulin - Humalog", "Insulin - Regular", "Jackson Pratt #1", "K Phos",
                      "KCL (Bolus)", "LR", "Lactate", "Lorazepam (Ativan)", "Lymphocytes", "MCH", "MCHC", "MCV",
                      "Magnesium", "Magnesium Sulfate", "Magnesium Sulfate (Bolus)", "Magnesium Sulfate Drug",
                      "Metoprolol", "Metoprolol Tartrate Drug", "Midazolam (Versed)", "Monocytes", "Morphine Sulfate",
                      "Neutrophils", "Nitroglycerin", "Norepinephrine", "OR Cell Saver Intake", "OR Crystalloid Intake",
                      "OR EBL", "Ostomy (output)", "PO Intake", "PT", "PTT", "Packed Red Blood Cells",
                      "Pantoprazole Drug", "Phenylephrine", "Phosphate", "Piggyback", "Platelet Count", "Potassium",
                      "Potassium Chloride", "Potassium Chloride Drug", "Pre-Admission", "RDW", "Red Blood Cells",
                      "Sodium", "Sodium Chloride 0.9%  Flush Drug", "Solution", "Specific Gravity", "Sterile Water",
                      "Stool Out Stool", "TF Residual", "Ultrafiltrate Ultrafiltrate", "Urea Nitrogen",
                      "Urine Out Incontinent", "Void", "White Blood Cells", "pCO2", "pH", "pO2"],
            "univariate": ["Glucose"],
            "multivariate": slice("Alanine Aminotransferase (ALT)", "pO2"),
            "timepoint": ['normalized_time'],
            "exogenous": ["normalized_time"],
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-IV-v3.1-ir": {
        "paths": [("{root}/disease/PhysioNet_MIMIC-IV_v3.1/02_multi_source_irregular.zip",
                   "02_multi_source_irregular/1min"),
                  ],
        "columns": {
            "names": ["Date", "normalized_time", "deltas",
                      "Acetaminophen-IV", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%",
                      "Alkaline Phosphatase", "Anion Gap", "Asparate Aminotransferase (AST)", "Base Excess",
                      "Basophils", "Bicarbonate", "Bilirubin, Total", "Calcium Gluconate", "Calcium, Total",
                      "Calculated Total CO2", "Cefazolin", "Cefepime", "Ceftriaxone", "Chloride", "Creatinine",
                      "Dexmedetomidine (Precedex)", "Dextrose 5%", "Emesis", "Eosinophils", "Famotidine (Pepcid)",
                      "Fecal Bag", "Fentanyl", "Fentanyl (Concentrate)", "Foley", "Free Water", "Furosemide (Lasix)",
                      "GT Flush", "Gastric Meds", "Glucose", "Hematocrit", "Hemoglobin", "Heparin Sodium",
                      "Heparin Sodium (Prophylaxis)", "Hydralazine", "Hydromorphone (Dilaudid)", "Insulin - Glargine",
                      "Insulin - Humalog", "Insulin - Regular", "K Phos", "KCL (Bolus)", "Lactate",
                      "Lorazepam (Ativan)", "Lymphocytes", "MCH", "MCV", "Magnesium", "Magnesium Sulfate",
                      "Magnesium Sulfate (Bolus)", "Metoprolol", "Metronidazole", "Midazolam (Versed)", "Monocytes",
                      "Morphine Sulfate", "Nasogastric", "Neutrophils", "Nitroglycerin", "Norepinephrine",
                      "OR Cell Saver Intake", "OR EBL", "OR Urine", "Oral Gastric", "PT", "PTT",
                      "Packed Red Blood Cells", "Pantoprazole (Protonix)", "Pantoprazole (Protonix) Continuous",
                      "Phenylephrine", "Phosphate", "Piggyback", "Piperacillin/Tazobactam (Zosyn)", "Platelet Count",
                      "Potassium", "Potassium Chloride", "Pre-Admission", "Propofol", "RDW", "Red Blood Cells",
                      "Sodium", "Solution", "Specific Gravity", "Sterile Water", "Stool", "Straight Cath",
                      "TF Residual", "TF Residual Output", "Urea Nitrogen", "Vancomycin", "Void", "White Blood Cells",
                      "pCO2", "pH", "pO2"],
            "univariate": ["Glucose"],
            "multivariate": slice("Acetaminophen-IV", "pO2"),
            "timepoint": ['normalized_time'],
            "exogenous": ["normalized_time"],
            "exogenous2": ["normalized_time"]
        }
    },

    # [Protein - Vaccine Response]
    # paper url: https://arxiv.org/abs/2505.14725
    "HR-VILAGE-3K3M-ir-pca": {
        "paths": [("{root}/../protein/HR-VILAGE-3K3M/02_multi_source_irregular(pca100).zip",
                   "02_multi_source_irregular(pca100)")],
        "columns": {
            "names": ['title', 'timepoint', 'standard_scaled_timepoint', 'responder2', 'responder'] + \
                     [f'PC{i}' for i in range(1, 101)],
            "timepoint": ["standard_scaled_timepoint"],  # indeed timepoint
            "univariate": [f'PC{i}' for i in range(1, 101)],
            "multivariate": [f'PC{i}' for i in range(1, 101)],
            "exogenous": ["timepoint"],
            "exogenous2": ["timepoint"],
        },
    },

    "HR-VILAGE-3K3M-ir": {
        "paths": [("{root}/../protein/HR-VILAGE-3K3M/02_multi_source_irregular(gene_expr).zip",
                   "02_multi_source_irregular(gene_expr)")],
        "columns": {
            "names": ['title', 'timepoint', 'deltas', 'standard_scaled_timepoint', 'responder2', 'responder'],
            "timepoint": ["deltas"],  # indeed timepoint
            "univariate": ["A1BG"],
            "multivariate": slice("A1BG", "ZZZ3"),
            "exogenous": ["standard_scaled_timepoint"],
            "exogenous2": ["deltas"],
        },
    },

    "HR-VILAGE-3K3M-ir-transpose": {
        "paths": [("{root}/../protein/HR-VILAGE-3K3M/02_multi_source_irregular(gene_expr+transpose).zip",
                   "02_multi_source_irregular(gene_expr+transpose)")],
        "columns": {
            "names": ['title', 'timepoint', 'deltas', 'standard_scaled_timepoint', 'responder2', 'responder'],
            "timepoint": ["deltas"],  # indeed timepoint
            "univariate": ["A1BG"],
            "multivariate": slice("A1BG", "ZZZ3"),
            "exogenous": ["timepoint"],
            "exogenous2": ["deltas"],
        },
    },

}

smir_metadata = dict()
smir_metadata.update(multi_source_irregular_metadata)


def prepare_smir_datasets(data_root: str,
                          dataset_name: str,
                          supervised_strategy: AbstractSupervisedStrategy,
                          split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                          device: Union[Literal['cpu', 'mps', 'cuda'], str] = 'cpu',
                          transpose: bool = False,
                          **task_kwargs: Dict[str, Any]) -> SMIrDatasetSequence:
    """

        Prepare several ``SMIrDataset`` for machine/incremental learning tasks.

    """

    assert dataset_name in smir_metadata, \
        f"Dataset '{dataset_name}' not found in metadata. The dataset name should be one of {list(smir_metadata.keys())}."

    given_metadata = smir_metadata[dataset_name]
    paths = given_metadata['paths']  # Paths are a list of paths or csv files

    for i, path in enumerate(paths):  # Update 'data_root' in paths
        if isinstance(path, str):
            paths[i] = os.path.normpath(path.format(root=data_root))
        elif isinstance(path, (tuple, list)) and len(path) == 2:
            paths[i] = (os.path.normpath(path[0].format(root=data_root)), path[1])

    task_ts = task_kwargs.get('ts', 'univariate')
    task_use_ex = task_kwargs.get('use_ex', False)
    task_ex_mask = task_kwargs.get('ex_ts_mask', False)
    task_use_ex2 = task_kwargs.get('use_ex2', False)
    task_shuffle = task_kwargs.get('shuffle', True)  # False: keep the file order for debugging
    task_show_loading_progress = task_kwargs.get('show_loading_progress', True)
    task_max_loading_workers = task_kwargs.get('max_loading_workers', 1)

    if task_ts not in given_metadata['columns']:
        raise ValueError(f"Label '{task_ts}' not found in dataset '{dataset_name}[columns]' metadata.")
    variables = given_metadata['columns'].get(task_ts, None)

    if 'timepoint' not in given_metadata['columns']:
        raise ValueError(f"Label 'timepoint' not found in dataset '{dataset_name}[columns]' metadata.")
    timepoint_variables = given_metadata['columns'].get('timepoint', None)

    ex_variables = given_metadata['columns'].get('exogenous', None) if task_use_ex else None
    ex2_variables = given_metadata['columns'].get('exogenous2', None) if task_use_ex2 else None

    # A path supports csv file(s), or csv file(s) in a directory.
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
        random.shuffle(filenames)  # shuffle filenames for better data distribution

    logging.getLogger().info('Loading (all or part of) {} files in {}'.format(len(filenames), paths))
    load_smt_args = {
        'filenames': filenames,
        'variables': variables,
        'timepoint_variables': timepoint_variables,
        'ex_variables': ex_variables,
        'mask_ex_variables': task_ex_mask,
        'ex2_variables': ex2_variables,
        'supervised_strategy': supervised_strategy,
        'split_ratios': split_ratios,
        'device': device,
        'transpose': transpose,
        'show_loading_progress': task_show_loading_progress,
        'max_loading_workers': task_max_loading_workers,
    }
    smir_datasets = load_smir_datasets(**load_smt_args)

    return smir_datasets
