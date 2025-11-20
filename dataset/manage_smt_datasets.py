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

from fast.data import SMTDataset
from fast.data.processing import load_smt_datasets, retrieve_files_in_zip, SMTDatasetSequence

single_source_metadata = {

    # [General_MTS] General time series datasets are available at: https://zenodo.org/records/15255776
    "ETTh1": {
        "paths": [("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTh1.csv")],
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ETTh2": {
        "paths": [("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTh2.csv")],
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ETTm1": {
        "paths": [("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTm1.csv")],
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ETTm2": {
        "paths": [("{root}/general_mts.zip", "general_mts/Github_ETT_small/ETTm2.csv")],
        "columns": {
            "univariate": ["OT"],
            "multivariate": ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "ExchangeRate": {
        "paths": [("{root}/general_mts.zip", "general_mts/Github_exchange_rate/exchange_rate.csv")],
        "columns": {
            "univariate": ["Singapore"],
            "multivariate": ["Australia", "British", "Canada", "Switzerland", "China", "Japan",
                             "New Zealand", "Singapore"],
            "exogenous2": ['day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "JenaClimate": {
        "paths": [("{root}/general_mts.zip", "general_mts/MaxPlanck_Jena_Climate/mpi_roof_2010a.csv")],
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
        "paths": [("{root}/general_mts.zip", "general_mts/UCI_Electricity/electricity_20160701_20190702.csv")],
        "columns": {
            "univariate": ["320"],
            "multivariate": slice("0", "320"),
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS03": {
        "paths": [("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems03_flow.csv")],
        "columns": {
            "univariate": ["357"],
            "multivariate": slice("0", "357"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS04": {
        "paths": [("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems04_flow.csv")],
        "columns": {
            "univariate": ["306"],
            "multivariate": slice("0", "306"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS07": {
        "paths": [("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems07_flow.csv")],
        "columns": {
            "univariate": ["882"],
            "multivariate": slice("0", "882"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "PeMS08": {
        "paths": [("{root}/general_mts.zip", "general_mts/US_PEMS_03_04_07_08/pems08_flow.csv")],
        "columns": {
            "univariate": ["169"],
            "multivariate": slice("0", "169"),
            "exogenous2": ['minute_of_hour', 'hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "Traffic": {
        "paths": [("{root}/general_mts.zip", "general_mts/US_PEMS_Traffic/traffic_20160701_20180702.csv")],
        "columns": {
            "univariate": ["861"],
            "multivariate": slice("0", "169"),
            "exogenous2": ['hour_of_day', 'day_of_week', 'day_of_month', 'day_of_year']
        }
    },
    "US_CDC_Flu": {
        "paths": [("{root}/general_mts.zip",
                   "general_mts/US_CDC_Flu_Activation_Level/US_regional_flu_level_20181004_20250322.csv")],
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
        "paths": [("{root}/general_mts.zip", "general_mts/US_CDC_ILI/US_National_ILI_1997_2025.csv")],
        "columns": {
            "univariate": ["ILITOTAL"],
            "multivariate": ["WEIGHTED ILI", "UNWEIGHTED ILI", "AGE 0-4", "AGE 5-24",
                             "AGE 65", "ILITOTAL", "NUM. OF PROVIDERS", "TOTAL PATIENTS"],
            "exogenous2": ['day_of_month', 'week_of_year']
        }
    },
    "WHO_JAPAN_ILI": {
        "paths": [("{root}/general_mts.zip", "general_mts/WHO_Japan_ILI/Japan_ILI_19961006_20250309.csv")],
        "columns": {
            "univariate": ["ILI_ACTIVITY"],
            "multivariate": ["AH1", "AH1N12009", "AH3", "AH5", "ANOTSUBTYPED", "INF_A", "BVIC", "BYAM",
                             "BNOTDETERMINED", "INF_B", "INF_ALL", "INF_NEGATIVE", "ILI_ACTIVITY"],
            "exogenous2": ['day_of_month', 'week_of_year']
        }
    },

}

multi_source_metadata = {

    # [Climate] Climate time series datasets are available from: https://www.osti.gov/biblio/1394920
    # https://github.com/edebrouwer/gru_ode_bayes/blob/master/gru_ode_bayes/datasets/Climate/small_chunked_sporadic.csv
    "USHCN": {
        "paths": [
            ("{root}/climate/US_Historical_Climatology_Network/02_multi_source_sparse.zip", "02_multi_source_sparse")],
        "columns": {
            "names": ['normalized_time', 'Snow', 'SnowDepth', 'Precipitation', 'T_max', 'T_min'],
            "univariate": ["T_max", "T_min"],
            "multivariate": ["Snow", "SnowDepth", "Precipitation", "T_max", "T_min"],
            "exogenous2": ["normalized_time"]
        }
    },

    # [Disease]
    "SH_diabetes": {
        "paths": [("{root}/disease/sh_diabetes/02_multi_source.zip", "02_multi_source/T1DM"),
                  ("{root}/disease/sh_diabetes/02_multi_source.zip", "02_multi_source/T2DM")],
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
            "univariate": ["CGM"],
            "multivariate": ["CGM"],  # univariate and multivariate are the same, change it if needed.
            "exogenous": ['Dietary intake', 'Bolus insulin', 'Basal insulin',
                          'Insulin dose s.c. id:0 dosage', 'Insulin dose s.c. id:1 dosage',
                          'Non-insulin id:0 dosage', 'Non-insulin id:1 dosage', 'Non-insulin id:2 dosage',
                          'Insulin dose i.v. id:0 dosage', 'Insulin dose i.v. id:1 dosage',
                          'Insulin dose i.v. id:2 dosage']  # these features are sparse
        }
    },

    # [Disease-ICU]
    "PhysioNet-sparse-1min": {
        "paths": [
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse.zip", "02_multi_source_sparse/1min/set-a"),
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse.zip", "02_multi_source_sparse/1min/set-b"),
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse.zip", "02_multi_source_sparse/1min/set-c"),
        ],
        "columns": {
            # some demographics are removed in the forecasting task: 'Age', 'Gender', 'Height', 'ICUType', 'Weight'
            "names": ["Time", "normalized_time", "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
                      "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
                      "MAP", "MechVent",  "Mechanical ventilation respiration", # （0:false, 1:true）机械通气呼吸（0:否，1:是）
                      "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
                      "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC"],
            "univariate": ["Glucose"],
            "multivariate": ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
                             "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP",
                             "MechVent",  "Mechanical ventilation respiration", # （0:false, 1:true）机械通气呼吸（0:否，1:是）
                             "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2",
                             "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT",
                             "Urine", "WBC"],
            "time": ['normalized_time'],
            "exogenous2": ["normalized_time"]
        }
    },

    "PhysioNet-sparse-1hour": {
        "paths": [
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse.zip", "02_multi_source_sparse/1hour/set-a"),
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse.zip", "02_multi_source_sparse/1hour/set-b"),
            ("{root}/disease/PhysioNet_Challenge_2012/02_multi_source_sparse.zip", "02_multi_source_sparse/1hour/set-c"),
        ],
        "columns": {
            # some demographics are removed in the forecasting task: 'Age', 'Gender', 'Height', 'ICUType', 'Weight'
            "names": ["Time", "normalized_time", "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
                      "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg",
                      "MAP", "MechVent",  "Mechanical ventilation respiration", # （0:false, 1:true）机械通气呼吸（0:否，1:是）
                      "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
                      "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC"],
            "univariate": ["Glucose"],
            "multivariate": ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol", "Creatinine",
                             "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K", "Lactate", "Mg", "MAP",
                             "MechVent",  "Mechanical ventilation respiration", # （0:false, 1:true）机械通气呼吸（0:否，1:是）
                             "Na", "NIDiasABP", "NIMAP", "NISysABP", "PaCO2", "PaO2",
                             "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT",
                             "Urine", "WBC"],
            "time": ['normalized_time'],
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-III-Ext-tPatchGNN-sparse-1min": {
        "paths": [
            ("{root}/disease/PhysioNet_MIMIC-III-Ext-tPatchGNN/02_multi_source_sparse.zip",
             "02_multi_source_sparse/1min"),
        ],
        "columns": {
            "names": ["Date", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%", "Alkaline Phosphatase",
                      "Anion Gap", "Asparate Aminotransferase (AST)", "Aspirin Drug", "Base Excess", "Basophils",
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
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-III-Ext-tPatchGNN-sparse-1hour": {
        "paths": [
            ("{root}/disease/PhysioNet_MIMIC-III-Ext-tPatchGNN/02_multi_source_sparse.zip",
             "02_multi_source_sparse/1hour"),
        ],
        "columns": {
            "names": ["Date", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%", "Alkaline Phosphatase",
                      "Anion Gap", "Asparate Aminotransferase (AST)", "Aspirin Drug", "Base Excess", "Basophils",
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
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-III-v1.4-sparse-1min": {
        "paths": [
            ("{root}/disease/PhysioNet_MIMIC-III_v1.4/02_multi_source_sparse.zip", "02_multi_source_sparse/1min"),
        ],
        "columns": {
            "names": ["Date", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%", "Alkaline Phosphatase",
                      "Anion Gap", "Asparate Aminotransferase (AST)", "Aspirin Drug", "Base Excess", "Basophils",
                      "Bicarbonate", "Bilirubin, Total", "Bisacodyl Drug", "Calcium Gluconate", "Calcium, Total",
                      "Calculated Total CO2", "Chest Tube #1", "Chest Tube #2", "Chloride", "Condom Cath", "Creatinine",
                      "D5 1/2NS", "D5W Drug", "Dextrose 5%", "Docusate Sodium Drug", "Eosinophils", "Fecal Bag",
                      "Foley",
                      "Furosemide (Lasix)", "GT Flush", "Gastric Gastric Tube", "Gastric Meds", "Glucose", "Hematocrit",
                      "Hemoglobin", "Heparin Sodium", "Humulin-R Insulin Drug", "Hydralazine", "Insulin - Glargine",
                      "Insulin - Humalog", "Insulin - Regular", "Jackson Pratt #1", "K Phos", "KCL (Bolus)", "LR",
                      "Lactate", "Lorazepam (Ativan)", "Lymphocytes", "MCH", "MCHC", "MCV", "Magnesium",
                      "Magnesium Sulfate", "Magnesium Sulfate (Bolus)", "Magnesium Sulfate Drug", "Metoprolol",
                      "Metoprolol Tartrate Drug", "Midazolam (Versed)", "Monocytes", "Morphine Sulfate", "Neutrophils",
                      "Nitroglycerin", "Norepinephrine", "OR Cell Saver Intake", "OR Crystalloid Intake", "OR EBL",
                      "Ostomy (output)", "PO Intake", "PT", "PTT", "Packed Red Blood Cells", "Pantoprazole Drug",
                      "Phenylephrine", "Phosphate", "Piggyback", "Platelet Count", "Potassium", "Potassium Chloride",
                      "Potassium Chloride Drug", "Pre-Admission", "RDW", "Red Blood Cells", "Sodium",
                      "Sodium Chloride 0.9%  Flush Drug", "Solution", "Specific Gravity", "Sterile Water",
                      "Stool Out Stool", "TF Residual", "Ultrafiltrate Ultrafiltrate", "Urea Nitrogen",
                      "Urine Out Incontinent", "Void", "White Blood Cells", "pCO2", "pH", "pO2"],
            "univariate": ["Glucose"],
            "multivariate": slice("Alanine Aminotransferase (ALT)", "pO2"),
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-III-v1.4-sparse-1hour": {
        "paths": [
            ("{root}/disease/PhysioNet_MIMIC-III_v1.4/02_multi_source_sparse.zip", "02_multi_source_sparse/1hour"),
        ],
        "columns": {
            "names": ["Date", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%", "Alkaline Phosphatase",
                      "Anion Gap", "Asparate Aminotransferase (AST)", "Aspirin Drug", "Base Excess", "Basophils",
                      "Bicarbonate", "Bilirubin, Total", "Bisacodyl Drug", "Calcium Gluconate", "Calcium, Total",
                      "Calculated Total CO2", "Chest Tube #1", "Chest Tube #2", "Chloride", "Condom Cath", "Creatinine",
                      "D5 1/2NS", "D5W Drug", "Dextrose 5%", "Docusate Sodium Drug", "Eosinophils", "Fecal Bag",
                      "Foley",
                      "Furosemide (Lasix)", "GT Flush", "Gastric Gastric Tube", "Gastric Meds", "Glucose", "Hematocrit",
                      "Hemoglobin", "Heparin Sodium", "Humulin-R Insulin Drug", "Hydralazine", "Insulin - Glargine",
                      "Insulin - Humalog", "Insulin - Regular", "Jackson Pratt #1", "K Phos", "KCL (Bolus)", "LR",
                      "Lactate", "Lorazepam (Ativan)", "Lymphocytes", "MCH", "MCHC", "MCV", "Magnesium",
                      "Magnesium Sulfate", "Magnesium Sulfate (Bolus)", "Magnesium Sulfate Drug", "Metoprolol",
                      "Metoprolol Tartrate Drug", "Midazolam (Versed)", "Monocytes", "Morphine Sulfate", "Neutrophils",
                      "Nitroglycerin", "Norepinephrine", "OR Cell Saver Intake", "OR Crystalloid Intake", "OR EBL",
                      "Ostomy (output)", "PO Intake", "PT", "PTT", "Packed Red Blood Cells", "Pantoprazole Drug",
                      "Phenylephrine", "Phosphate", "Piggyback", "Platelet Count", "Potassium", "Potassium Chloride",
                      "Potassium Chloride Drug", "Pre-Admission", "RDW", "Red Blood Cells", "Sodium",
                      "Sodium Chloride 0.9%  Flush Drug", "Solution", "Specific Gravity", "Sterile Water",
                      "Stool Out Stool", "TF Residual", "Ultrafiltrate Ultrafiltrate", "Urea Nitrogen",
                      "Urine Out Incontinent", "Void", "White Blood Cells", "pCO2", "pH", "pO2"],
            "univariate": ["Glucose"],
            "multivariate": slice("Alanine Aminotransferase (ALT)", "pO2"),
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-IV-v3.1-sparse-1min": {
        "paths": [
            ("{root}/disease/PhysioNet_MIMIC-IV_v3.1/02_multi_source_sparse.zip", "02_multi_source_sparse/1min"),
        ],
        "columns": {
            "names": ["Date", "Acetaminophen-IV", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%",
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
            "exogenous2": ["normalized_time"]
        }
    },

    "MIMIC-IV-v3.1-sparse-1hour": {
        "paths": [
            ("{root}/disease/PhysioNet_MIMIC-IV_v3.1/02_multi_source_sparse.zip", "02_multi_source_sparse/1hour"),
        ],
        "columns": {
            "names": ["Date", "Acetaminophen-IV", "Alanine Aminotransferase (ALT)", "Albumin", "Albumin 5%",
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
            "exogenous2": ["normalized_time"]
        }
    },

    # [Energy - power load]
    "SouthPT-sparse-1day": {  # sparse
        "paths": [
            ("{root}/energy_electronic_power_load/SouthernChina_transformer_power_load_2021_2023/02_multi_source_sparse.zip",
             "02_multi_source_sparse/[downsample]1day")],
        "columns": {
            "univariate": ["LOAD"],
            "multivariate": ["LOAD"],
            "exogenous": ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP'],
            "exogenous2": ['day_of_week', 'day_of_month', 'day_of_year'],
        },
    },

    # [Energy - solar]
    "GreecePV-1day": {
        "paths": [("{root}/energy_solar/Github_Greece_solar_energy_forecasting_2017_2020/02_multi_source.zip",
                   "02_multi_source/[downsample]1day")],
        "columns": {
            "univariate": ["Solar energy (MW)"],
            "multivariate": ["Solar energy (MW)"],  # all target values of several time series is the same here
            "exogenous": ["airTemperature", "cloudCover", "gust", "humidity", "precipitation", "pressure",
                          "visibility", "windDirection", "windSpeed"]
        }
    },

    "ods-sparse-1day": {
        "paths": [("{root}/energy_solar/opendata_ods032/02_multi_source_sparse.zip",
                   "02_multi_source_sparse/[downsample]1day")],
        "columns": {
            "names": ["Date", "region_id", "Measured & Upscaled", "Most recent forecast", "Most recent P10",
                      "Most recent P90",
                      "Day Ahead 11AM forecast", "Day Ahead 11AM P10", "Day Ahead 11AM P90", "Day-ahead 6PM forecast",
                      "Day-ahead 6PM P10", "Day-ahead 6PM P90", "Week-ahead forecast", "Week-ahead P10",
                      "Week-ahead P90", "Monitored capacity", "Load factor"],
            "univariate": ["Measured & Upscaled"],
            "multivariate": ["Measured & Upscaled"],
            "exogenous": ["Most recent forecast", "Most recent P10", "Most recent P90",
                          "Day Ahead 11AM forecast", "Day Ahead 11AM P10", "Day Ahead 11AM P90",
                          "Day-ahead 6PM forecast", "Day-ahead 6PM P10", "Day-ahead 6PM P90",
                          "Week-ahead forecast", "Week-ahead P10", "Week-ahead P90",
                          "Monitored capacity", "Load factor"],
            "exogenous2": ["day_of_week", "day_of_month", "day_of_year"]
        }
    },

    "pvod-1day": {
        "paths": [("{root}/energy_solar/solener_pvod_2018_2019/02_multi_source.zip",
                   "02_multi_source/[downsample+dropna]1day")],
        "columns": {
            "names": ["Date", "station_id", "nwp_globalirrad", "nwp_directirrad", "nwp_temperature", "nwp_humidity",
                      "nwp_windspeed", "nwp_winddirection", "nwp_pressure", "lmd_totalirrad", "lmd_diffuseirrad",
                      "lmd_temperature", "lmd_pressure", "lmd_winddirection", "lmd_windspeed", "power"],
            "univariate": ["power"],
            "multivariate": ["power"],
            "exogenous": ["nwp_globalirrad", "nwp_directirrad", "nwp_temperature", "nwp_humidity",
                          "nwp_windspeed", "nwp_winddirection", "nwp_pressure", "lmd_totalirrad", "lmd_diffuseirrad",
                          "lmd_temperature", "lmd_pressure", "lmd_winddirection", "lmd_windspeed"]
        }
    },

    # [Energy - smart meter]
    "GoiEner-1day": {
        "paths": [("{root}/energy_smart_meter/Spanish_GoiEner_smart_meters_data_2014_2022/02_multi_source_imputed.zip",
                   "02_multi_source_imputed/[downsample]1day")],
        "columns": {
            "names": ["Date", "kWh", "imputed"],
            "univariate": ["kWh"],
            "multivariate": ["kWh"],
        }
    },

    # [Energy - wind]
    "GreeceWPF": {
        "paths": [("{root}/energy_wind/GitHub_Greece_wind_energy_forecasting_2017_2020/02_multi_source.zip",
                   "02_multi_source/1hour")],
        "freq": ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "univariate": ["Wind energy (MW)"],
            "multivariate": ["Wind energy (MW)"],  # all target values of several time series is the same here
            "exogenous": ["airTemperature", "cloudCover", "gust", "humidity", "precipitation", "pressure",
                          "visibility", "windDirection", "windSpeed"]
        }
    },

    "GreeceWPF-1day": {
        "paths": [("{root}/energy_wind/GitHub_Greece_wind_energy_forecasting_2017_2020/02_multi_source.zip",
                   "02_multi_source/1day")],
        "columns": {
            "univariate": ["Wind energy (MW)"],
            "multivariate": ["Wind energy (MW)"],  # all target values of several time series is the same here
            "exogenous": ["airTemperature", "cloudCover", "gust", "humidity", "precipitation", "pressure",
                          "visibility", "windDirection", "windSpeed"]
        }
    },

    "SDWPF": {
        "paths": [
            ("{root}/energy_wind/KDDCup2022_Spatial_Dynamic_Wind_Power_Forecasting/02_multi_source_szw(backup).zip",
             "02_multi_source_szw(backup)/30min")],
        "freq": ["10min", "30min", "1hour", "6hour", "12hour", "1day"],
        "columns": {
            "univariate": ["Patv"],
            "multivariate": ["Patv"],
            "exogenous": ["Wspd", "Wdir", "Etmp", "Itmp", "Ndir", "Pab1", "Pab2", "Pab3", "Prtv"]
        }
    },

    "WSTD2": {
        "paths": [("{root}/energy_wind/Zenodo_Wind_Spatio_Temporal_Dataset2_2010_2011/02_multi_source_sparse.zip",
                   "02_multi_source_sparse/1hour")],
        "freq": ["1hour", "6hour", "12hour", "1day"],
        "columns": {
            "univariate": ["Power"],
            "multivariate": ["Power"],
        }
    },

    # [Spatio-temporal] Human activity recognition datasets, the "time" feature is normalized.
    # The original data is from: https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity
    "HumanActivity": {
        "paths": [("{root}/../spatio_temporal/human_activity/02_multi_source_sparse.zip", "02_multi_source_sparse")],
        "columns": {
            "univariate": ["010-000-024-033_x", "010-000-024-033_y", "010-000-024-033_z"],
            "multivariate": ["010-000-024-033_x", "010-000-024-033_y", "010-000-024-033_z",
                             "010-000-030-096_x", "010-000-030-096_y", "010-000-030-096_z",
                             "020-000-032-221_x", "020-000-032-221_y", "020-000-032-221_z",
                             "020-000-033-111_x", "020-000-033-111_y", "020-000-033-111_z"],
            "exogenous2": ["normalized_time"],
        }
    },

}

protein_pKa_metadata = {
    # [Protein] pKa prediction datasets: http://computbiophys.com/DeepKa/database
    "phmd_2d_549_train": {
        "paths": [("{root}/../protein/pKa/phmd_2d_549_sparse.zip", "phmd_2d_549_sparse/train")],
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
        "paths": [("{root}/../protein/pKa/phmd_2d_549_sparse.zip", "phmd_2d_549_sparse/valid")],
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
        "paths": [("{root}/../protein/pKa/phmd_2d_549_sparse.zip", "phmd_2d_549_sparse/test")],
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
}

simulation_metadata = {
    "GFM": {
        "paths": [("{root}/simulation/gfm_sim_abc.zip", "gfm_sim_abc")],
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
    },

    "DIgSILENT": {
        # "paths": ["{root}/simulation/DIgSILENT PowerFactory Eigenvalue Prediction/model_analysis"],
        "paths": [("{root}/simulation/[raw]DIgSILENT PowerFactory Eigenvalue Prediction (samples).zip", "")],
        "columns": {
            "names": ["Time", "G2 P", "G2 Q", "G2 speed", "G3 P", "G3 Q", "G3 speed", "G1 P", "G1 Q", "G1 speed",
                      "Bus 1 Vm", "Bus 1 V_angle", "Bus 2 Vm", "Bus 2 V_angle", "Bus 8 Vm", "Bus 8 V_angle",
                      "Bus 9 Vm", "Bus 9 V_angle", "Bus 3 Vm", "Bus 3 V_angle", "Bus 6 Vm", "Bus 6 V_angle", "Bus 4 Vm",
                      "Bus 4 V_angle", "Bus 5 Vm", "Bus 5 V_angle", "Bus 7 Vm", "Bus 7 V_angle", "Line 5-7 Im",
                      "Line 5-7 I_angle", "Line 7-8 Im", "Line 7-8 I_angle", "Line 8-9 Im", "Line 8-9 I_angle",
                      "Line 6-9 Im", "Line 6-9 I_angle", "Line 4-6 Im", "Line 4-6 I_angle", "Line 4-5 Im",
                      "Line 4-5 I_angle", "EV_0_real", "EV_0_imag", "EV_1_real", "EV_1_imag", "EV_2_real", "EV_2_imag",
                      "EV_3_real", "EV_3_imag", "EV_4_real", "EV_4_imag", "EV_5_real", "EV_5_imag", "EV_6_real",
                      "EV_6_imag", "EV_7_real", "EV_7_imag", "EV_8_real", "EV_8_imag", "EV_9_real", "EV_9_imag",
                      "EV_10_real", "EV_10_imag", "EV_11_real", "EV_11_imag", "EV_12_real", "EV_12_imag",
                      "EV_13_real", "EV_13_imag", "EV_14_real", "EV_14_imag", "EV_15_real", "EV_15_imag",
                      "EV_16_real", "EV_16_imag", "EV_17_real", "EV_17_imag", "EV_18_real", "EV_18_imag"],
            "univariate": ["EV_0_real", "EV_0_imag"],
            "multivariate": slice("EV_0_real", "EV_9_imag"),
            "exogenous": slice("G2 P", "Line 4-5 I_angle"),
            "exogenous2": slice("G2 P", "Line 4-5 I_angle"),
        },
    }
}

smt_metadata = dict()
smt_metadata.update(single_source_metadata)
smt_metadata.update(multi_source_metadata)
smt_metadata.update(protein_pKa_metadata)
smt_metadata.update(simulation_metadata)


def prepare_smt_datasets(data_root: str,
                         dataset_name: str,
                         input_window_size: int = 96,
                         output_window_size: int = 24,
                         horizon: int = 1,
                         stride: int = 1,
                         split_ratios: Union[int, float, Tuple[float, ...], List[float]] = None,
                         split_strategy: Literal['intra', 'inter'] = 'intra',
                         device: Union[Literal['cpu', 'mps', 'cuda'], str] = 'cpu',
                         **task_kwargs: Dict[str, Any]) -> SMTDatasetSequence:
    """
        Prepare several ``SMTDataset`` for machine/incremental learning tasks.

        The default **float type** is ``float32``, you can change it in ``load_smt_datasets()`` to ``float64`` if needed .

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
                            ``ts_mask``:     bool, whether to mask the time series variables, default is False.
                            ``use_ex``: bool, whether to use exogenous variables, default is None.
                            ``ex_ts_mask``: bool, whether to mask the exogenous variables, default is False.
                            ``use_ex2``: bool, whether to use time features, default is False.
                            ``shuffle``: whether to shuffle the data files. Default is False (for debugging).
                            ``show_loading_progress``: bool, whether to show the loading progress bar, default is True.
                            ``max_loading_workers``: int, the maximum number of workers for data loading, default is 1.

        :return: the (split) datasets as ``SMTDataset`` objects or ``SMDDataset`` object.
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
    task_ts_mask = task_kwargs.get('ts_mask', False)
    task_use_ex = task_kwargs.get('use_ex', False)
    task_ex_mask = task_kwargs.get('ex_ts_mask', False)
    task_ex2 = task_kwargs.get('use_ex2', False)
    task_shuffle = task_kwargs.get('shuffle', True)  # False: keep the file order for debugging
    task_show_loading_progress = task_kwargs.get('show_loading_progress', True)
    task_max_loading_workers = task_kwargs.get('max_loading_workers', 1)

    variables = given_metadata['columns'].get(task_ts, None)
    if variables is None:
        raise ValueError(f"Task type '{task_ts}' not found in dataset '{dataset_name}' metadata.")

    ex_variables = given_metadata['columns'].get('exogenous', None) if task_use_ex else None
    ex2_variables = given_metadata['columns'].get('exogenous2', None) if task_ex2 else None

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

    logging.getLogger().info('Loading {} files in {}'.format(len(filenames), paths))
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
        'show_loading_progress': task_show_loading_progress,
        'max_loading_workers': task_max_loading_workers,
    }
    smt_datasets = load_smt_datasets(**load_smt_args)

    return smt_datasets


def verify_smt_datasets():
    """
        Verify the ``SMTDataset`` metadata.
    """
    data_root = os.path.expanduser('~/data/time_series') if os.name == 'posix' else 'D:/data/time_series'

    ds_names = list(multi_source_metadata.keys())
    for i, name in enumerate(ds_names):
        task_config = {'ts': 'multivariate', 'ts_mask': True, 'use_ex': True, 'ex_ts_mask': True, 'use_ex2': True}
        print(i, end='\t')
        datasets = prepare_smt_datasets(data_root, name, 48, 24, 1, 1, **task_config)
        datasets = [datasets] if isinstance(datasets, SMTDataset) else datasets
        print('\n'.join([str(ds) for ds in datasets]))
