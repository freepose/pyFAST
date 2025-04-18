#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data

from typing import Literal


def supervised_task(train_ds: data.Dataset,
                    val_ds: data.Dataset,
                    model_cls: nn.Module,
                    model_params: dict,
                    device: torch.device = None,
                    torch_float_type=torch.float32,
                    optimizer=None, lr_scheduler=None,
                    criterion=None, evaluator=None,
                    scaler=None, ex_scaler=None,
                    epoch_range: tuple[int] = (1, 10),
                    batch_size: int = 32,
                    shuffle: bool = True,
                    verbose: bool = False,
                    display_interval: int = None):
    """

        Dataset Parameters

        Model Parameters

        Trainer parameters

    """
    pass