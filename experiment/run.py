#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Literal, Dict, Union, List, Tuple, Any

from fast import get_common_params, get_device
from fast.data import StandardScale, scale_several_time_series, SSTDataset, SMTDataset
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE, AbstractMetric

from fast.model.base import get_model_info, covert_parameters


def supervised_learning(model_and_params: Union[Tuple[nn.Module, dict], List[Tuple[nn.Module, dict]]],
                        datasets: Union[Union[SSTDataset, SMTDataset], Tuple[Union[SSTDataset, SMTDataset], ...]],
                        device: str = 'cpu',
                        lr: float = 0.0001, scheduler_step_size: int = 15, scheduler_gamma: float = 0.996,
                        stopper_patience: int = 7, stopper_mode: Literal['abs', 'rel'] = 'rel',
                        stopper_delta: float = 0.01,
                        criterion: str = 'MSE',
                        metrics: Union[List[str], Tuple[str]] = ('MSE', 'MAE'),
                        max_epochs: int = 500, batch_size: int = 32, shuffle: bool = True,
                        verbose: Literal[0, 1, 2] = 1,
                        scale_on: Literal['ts', 'ex_ts', 'both'] = None):
    """
        Supervised learning training function.
        :param model_and_params:
        :param datasets:
        :param device:
        :param lr:
        :param scheduler_step_size:
        :param scheduler_gamma:
        :param stopper_patience:
        :param stopper_mode:
        :param stopper_delta:
        :param criterion:
        :param metrics:
        :param max_epochs:
        :param batch_size:
        :param shuffle:
        :param verbose:
        :param scale_on:
    :return:
    """

    train_ds, val_ds, test_ds = [*datasets, None, None][:3]
    assert train_ds is not None, 'train_ds is None'

    scaler, ex_scaler = None, None
    if scale_on in ['ts', 'both']:
        scaler = scale_several_time_series(StandardScale(), train_ds.ts, train_ds.ts_mask)
    if scale_on in ['ex_ts', 'both'] and train_ds.ex_ts is not None:
        ex_scaler = scale_several_time_series(StandardScale(), train_ds.ex_ts, train_ds.ex_ts_mask)

    model_class, model_params = model_and_params
    common_ds_params = get_common_params(model_class.__init__, train_ds.__dict__)  # Adapt dataset parameters
    model_settings = {**common_ds_params, **model_params}
    model = model_class(**model_settings)

    model = covert_parameters(model, torch.float32)
    print(get_model_info(model))

    model_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_weights, lr=lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    evaluator = Evaluator(metrics)
    criterion = evaluator.available_metrics[criterion]()  # Instantiate the criterion

    trainer = Trainer(get_device(device), model, is_initial_weights=True, is_compile=False,
                      optimizer=optimizer, lr_scheduler=lr_scheduler,
                      stopper=EarlyStop(stopper_patience, stopper_delta, stopper_mode),
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)
    print(trainer)

    history = trainer.fit(train_ds, val_ds,
                          epoch_range=(1, max_epochs), batch_size=batch_size, shuffle=shuffle,
                          verbose=verbose)

    if test_ds is not None:
        results = trainer.evaluate(test_ds, batch_size, None, False, is_online=False)
        print('test', results)
    elif val_ds is not None:
        results = trainer.evaluate(val_ds, batch_size, None, False, is_online=False)
        print('val', results)

    return trainer, history
