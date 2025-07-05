#!/usr/bin/env python
# encoding: utf-8
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Literal, Union, List, Tuple

from experiment import load_sst_dataset, DotDict

from fast import get_common_kwargs, get_device, initial_seed, initial_logger
from fast.data import StandardScale, scale_several_time_series, SSTDataset, SMTDataset
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator

from fast.model.base import get_model_info, covert_parameters
from fast.model import ts_model_classes


def run_experiment(global_settings: DotDict, dataset_arguments: DotDict, trainer_arguments: DotDict,
                   model_and_arguments: List | Tuple, device: str = None, log_file: str = sys.stdout):
    """
        Run the experiment with the given dataset, trainer, and model settings.
        :param global_settings: the global settings, including seed, device, loss, metrics, log file and et al.
        :param dataset_arguments: dataset arguments
        :param trainer_arguments: trainer arguments
        :param model_and_arguments: model name and model arguments
        :param device: the device to run the experiment.
        :param log_file: the log file

        :return: training history, which is a dictionary containing the training and validation loss and metrics.
    """
    initial_seed(getattr(global_settings, 'seed', 2025))
    logger = initial_logger(log_file, logging.INFO)

    sst_datasets = load_sst_dataset(**dataset_arguments)
    train_ds, val_ds, test_ds = [*sst_datasets, None, None][:3]
    assert train_ds is not None, 'train dataset is None'

    logger.info('\n'.join([str(i) for i in (sst_datasets if len(sst_datasets) > 1 else [sst_datasets])]))

    model_name, settings, *update_args = model_and_arguments    # TODO: update args
    if model_name not in ts_model_classes:
        raise ValueError(f"Model '{model_name}' not found in ts_model_classes.")
    model_class = ts_model_classes[model_name]

    common_ds_args = get_common_kwargs(model_class.__init__, train_ds.__dict__)  # Adapt dataset parameters
    settings = {**common_ds_args, **settings}
    model = model_class(**settings)

    model = covert_parameters(model, torch.float32)
    logger.info(get_model_info(model))

    optim_args = getattr(trainer_arguments, 'optimizer', {})
    optimizer = optim.Adam(model.parameters(), **optim_args)
    scheduler_args = getattr(trainer_arguments, 'lr_scheduler', {'step_size': 10})
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_args)

    metrics = getattr(trainer_arguments, 'metrics', ["MSE", "MAE"])
    evaluator = Evaluator(metrics)

    criterion_name = getattr(global_settings, 'loss', 'MSE')
    criterion = evaluator.available_metrics[criterion_name]()  # Instantiate the criterion

    early_stop_args = getattr(trainer_arguments, 'early_stop', None)
    early_stop = EarlyStop(**early_stop_args) if early_stop_args else None

    scaler, ex_scaler = None, None
    if 'scale' in global_settings:
        if global_settings['scale'] in ['ts', 'both']:
            scaler = scale_several_time_series(StandardScale(), train_ds.ts, train_ds.ts_mask)

        if global_settings['scale'] in ['ex_ts', 'both'] and train_ds.ex_ts is not None:
            ex_scaler = scale_several_time_series(StandardScale(), train_ds.ex_ts, train_ds.ex_ts_mask)

    trainer = Trainer(get_device(device), model, is_initial_weights=True, is_compile=False,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=early_stop,
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)

    logger.info(trainer)

    fit_args = {k: v for k, v in trainer_arguments.items() if k not in ['optimizer', 'lr_scheduler', 'early_stop']}
    history = trainer.fit(train_ds, val_ds, **fit_args)

    batch_size = trainer_arguments.get('batch_size', 32)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, batch_size, None, False, is_online=False)
        logger.info('test {}'.format(results))
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, batch_size, None, False, is_online=False)
        logger.info('val {}'.format(results))

    return history