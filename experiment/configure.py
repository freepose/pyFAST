#!/usr/bin/env python
# encoding: utf-8

"""
    Read the configuration (json) file and prepare instantiated classes for experiments.
"""

import os, json

from typing import Dict, Any, Union, List, Tuple


class DotDict(Dict):
    """
        A dictionary that allows dot notation access to its keys.
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def dict_to_dotdict(obj: Union[Dict, List]):
    """
        Recursively convert a dictionary to a DotDict.
        :param obj: the object to convert.
        :return: the converted object.
    """

    if isinstance(obj, dict):
        return DotDict({k: dict_to_dotdict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [dict_to_dotdict(item) for item in obj]
    return obj


def load_json_as_dotdict(filename) -> DotDict:
    """
        Load the configuration file.
        :param filename: the name of the configuration file.
        :return: the configuration dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Configuration json file {filename} not found.")

    with open(filename, 'r') as f:
        data = json.load(f)
    return dict_to_dotdict(data)


class Configurator:
    """
        The ``Configurator`` class is responsible for loading and parsing the configuration file.
        Initialize the configurator with a configuration file and data root directory.

        :param filename: the path to the configuration json file.
        :param data_root: the root directory of the dataset (a csv file).
    """

    def __init__(self, filename: str = "./config_json/ett.json", data_root: str = None):

        self.data_root = data_root  # the root directory of the dataset (a csv file)
        self.configurations = load_json_as_dotdict(filename)

    def get_global(self) -> DotDict[str, Any]:
        """
            Get global settings from the configuration file.
            :return: the dictionary of global settings.
        """
        if 'global' not in self.configurations:
            raise ValueError("No global settings found in the configuration file.")

        global_settings = self.configurations['global']
        return global_settings

    def _check_labels(self, dataset: str, experiment: str) -> None:
        """
            Check if the labels are available in the configuration file.

            :param dataset: the name of the dataset.
            :param experiment: the name of the experiment.
            :return: True if the labels are available, False otherwise.
        """
        if dataset not in self.configurations:
            raise ValueError(f"Dataset {dataset} not found in the configuration file.")
        ds = self.configurations[dataset]

        if "meta" not in ds:
            raise ValueError(f"No meta information found for dataset {dataset} in the configuration file.")
        meta = ds['meta']

        if 'path' not in meta:
            raise ValueError(f" 'path' not found for dataset {dataset} in the configuration file.")

        if 'experiments' not in ds:
            raise ValueError(f"No experimental settings found for dataset {dataset} in the configuration file.")
        experiments = ds['experiments']

        if experiment not in experiments:
            raise ValueError(f"Experiment {experiment} not found in the '{dataset}.experiments' in the configuration file.")
        exp = ds.experiments[experiment]

        if 'targets' not in exp:
            raise ValueError(f"No targets label not found for '{dataset}.experiments.{experiment}' in the configuration file.")
        targets = exp.targets

        if targets not in meta:
            raise ValueError(f"Targets {targets} not found in '{dataset}.meta' in the configuration file.")

        if 'split' not in exp:
            raise ValueError(f"No dataset split information found for '{dataset}.experiments.{experiment}' in the configuration file.")

        if 'trainer' not in exp:
            raise ValueError(f"No trainer information found for '{dataset}.experiments.{experiment}' in the configuration file.")

    def get(self, dataset: str, experiment: str) -> Tuple[Dict[str, Any] | List, ...]:
        """
            Get arguments list, they ``load_sst_dataset`` arguments, ``Trainer`` arguments, and list of [model_name, model_arguments].

            (1) The keywords for ``load_sst_dataset`` function are needed:

                self.configurations[dataset].meta.path
                self.configurations[dataset].experiments[experiment].targets
                self.configurations[dataset].experiments[experiment].split

            (2) The keywords for ``Trainer`` class are needed:

                self.configurations[dataset].experiments[experiment].trainer

            (3) The keywords for models to be conducted are needed:

                self.configurations[dataset].experiments[experiment].models

            :param dataset: the name of the dataset.
            :param experiment: the name of the experiment.
            :return: (sst_dataset_arguments, trainer_arguments, models_arguments)
        """

        self._check_labels(dataset, experiment)

        filename = self.configurations[dataset].meta.path.format(root=self.data_root)
        meta = self.configurations[dataset].meta

        exp = self.configurations[dataset].experiments[experiment]
        target_variables = meta[exp.targets]

        sst_dataset_arguments = dict_to_dotdict({'filename': filename, 'variables': target_variables, **exp.split})

        return sst_dataset_arguments, exp.trainer, exp.models
