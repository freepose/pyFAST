#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from typing import Literal, Dict, Union

from fast import get_common_params
from fast.data import StandardScale, scale_several_time_series, SSTDataset, SMTDataset
from fast.train import Trainer
from fast.stop import EarlyStop
from fast.metric import Evaluator, MSE
from fast.model.base import count_parameters, covert_parameters


def supervised_learning(model_class: nn.Module,
                        model_params: Dict[str, any],
                        train_ds: Union[SSTDataset, SMTDataset],
                        val_ds: Union[SSTDataset, SMTDataset] = None,
                        test_ds: Union[SSTDataset, SMTDataset] = None,
                        device: torch.device = None,
                        torch_float_type=torch.float32,
                        scale_on: Literal['ts', 'ex_ts', 'both'] = None):

    scaler, ex_scaler = None, None
    if scale_on in ['ts', 'both']:
        scaler = scale_several_time_series(StandardScale(), train_ds.ts, train_ds.ts_mask)
    elif scale_on == ['ex_ts', 'both']:
        ex_scaler = scale_several_time_series(StandardScale(), train_ds.ex_ts, train_ds.ex_ts_mask)

    common_ds_params = get_common_params(model_class.__init__, train_ds.__dict__)
    model_settings = {**common_ds_params, **model_params}
    model = model_class(**model_settings)

    print('{}\n{}\n{}'.format(train_ds, val_ds, model))

    model_name = type(model).__name__
    model = covert_parameters(model, torch_float_type)
    print(model_name, count_parameters(model))

    model_weights = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model_weights, lr=0.0005, weight_decay=0.)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.996)

    criterion = MSE()
    evaluator = Evaluator(['MAE'])

    trainer = Trainer(device, model, is_initial_weights=True, is_compile=False,
                      optimizer=optimizer, lr_scheduler=lr_scheduler, stopper=EarlyStop(7),
                      criterion=criterion, evaluator=evaluator,
                      scaler=scaler, ex_scaler=ex_scaler)

    trainer.fit(train_ds, val_ds,
                epoch_range=(1, 500), batch_size=32, shuffle=True,
                verbose=True)

    if test_ds:
        val_results = trainer.evaluate(val_ds, 32, None, None, is_online=False)
        print(val_results)
