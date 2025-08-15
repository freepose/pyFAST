#!/usr/bin/env python
# encoding: utf-8

"""
    This package supports training all models, which includes sparse data and mask models in terms of uts and mts.
    Single computer/server version.
"""

import logging
import os, sys, platform
from typing import Literal, Tuple, List, Union, Dict, Callable

import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from .model.base import to_string, init_weights
from .metric import AbstractMetric, MSE
from .metric import AbstractEvaluator, EmptyEvaluator
from .data import AbstractScale, AbstractMask

class Trainer:
    """
        Trainer for large-scale time series dataset (``SSTDataset`` or ``SMTDataset``) training and evaluation.

        (1) ``SSTDataset`` works with a large-scale time series dataset.
            ``SMTDataset`` works with several large-scale time series datasets.

        (2) The metrics work with streaming aggregation of performance of batches.

        :param device: the device for model training.
        :param model: the model to be trained.
        :param is_initial_weights: whether to initialize the model weights.
        :param is_compile: whether to compile the model. The feature tor torch>=2.0.0
        :param optimizer: the optimizer to be used. If None, Adam optimizer will be used.
        :param lr_scheduler: the learning rate scheduler to be used. If None, StepLR will be used.
        :param stopper: the early stopper to be used. If None, no early stopping will be used.
        :param criterion: the loss function to be used. Default is ``MSE``.
        :param evaluator: the evaluator to be used. Default is ``EmptyEvaluator``, i.e., not evaluating.
        :param scaler: the scaler of time series of target variables. Default is ``None``.
        :param ex_scaler: the external scaler of time series of exogenous variables. Default is ``None``.
        :param impute_mask: the mask for missing values imputation. Default is ``None``.
                            Use this when pretrain/train the model with missing values imputation.
    """

    def __init__(self, device: torch.device,
                 model: nn.Module, is_initial_weights: bool = False, is_compile: bool = False,
                 optimizer=None, lr_scheduler=None, stopper=None,
                 criterion: AbstractMetric = MSE(), evaluator: AbstractEvaluator = None,
                 scaler: AbstractScale = None, ex_scaler: AbstractScale = None):

        self.device = device

        self.model = model
        self.is_initial_weights = is_initial_weights
        self.is_compile = is_compile

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.stopper = stopper

        self.criterion = criterion
        self.evaluator = evaluator if evaluator else EmptyEvaluator()

        self.scaler = scaler
        self.ex_scaler = ex_scaler

        if self.is_initial_weights:
            model.apply(init_weights)

        self.initialize_device()

        if self.is_compile:
            # MPS device may not support for this compiling.
            self.model = torch.compile(self.model)

        if self.optimizer is None:
            model_params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(model_params, lr=0.0001)

        if self.lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def initialize_device(self):
        """ Initialize model accelerator. """

        if self.device is not None: # reset all the model parameters to the device
            self.model = self.model.to(self.device)

        if self.device.type == 'cpu':
            if platform.machine() in ('x86_64', 'AMD64'):
                torch.set_num_threads(os.cpu_count() - 2)
            elif platform.machine() == 'arm64':
                torch.set_num_threads(os.cpu_count())
        elif self.device.type == 'cuda':
            cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            visible_gpus = [int(d.strip()) for d in cuda_visible_devices.split(',') if d.strip()]
            if len(visible_gpus) > 1 and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model, device_ids=visible_gpus)
        elif self.device.type == 'mps':
            pass

    def run_epoch(self, dataloader,
                  mode: Literal['train', 'val', 'online'] = 'train',
                  progress_status: str = None,
                  forcast_mask: AbstractMask = None,
                  impute_mask: AbstractMask = None) -> Dict[str, float]:
        """
            Execute one training/validation epoch.

            Note:
                - Automatic data scaling is applied if scalers are configured
                - Device transfer is handled automatically
                - Dynamic masking only applies during training mode

            :param dataloader: DataLoader containing batched data
            :param mode: Execution mode - 'train' enables gradients, 'val'/'online' for evaluation
            :param progress_status: Description text for progress bar, None disables progress bar
            :param forcast_mask: Dynamic mask strategy for **forecasting** tasks during training
            :param impute_mask: Dynamic mask strategy for **imputation** tasks during training

            :return: Dict containing epoch metrics like loss and evaluation scores

            :raises RuntimeError: If model forward pass fails
                    ValueError: If incompatible mask strategies are provided
        """

        self.model.train() if mode in ['train', 'online'] else self.model.eval()

        pbar = None
        if progress_status is not None:
            pbar = tqdm(total=len(dataloader), leave=False, file=sys.stdout)
            pbar.set_description(progress_status)

        self.criterion.reset()
        self.evaluator.reset()
        for batch_inputs, batch_outputs in dataloader:
            num_target_vars = len(batch_outputs)    # number of target variables
            num_exogenous_vars = len(batch_inputs) - num_target_vars    # number of exogenous variables

            if self.scaler is not None:
                batch_inputs[0] = self.scaler.transform(*batch_inputs[:num_target_vars])
                batch_outputs[0] = self.scaler.transform(*batch_outputs[:num_target_vars])

            if num_exogenous_vars > 0 and (self.ex_scaler is not None):
                batch_inputs[num_target_vars] = self.ex_scaler.transform(batch_inputs[num_target_vars])

            if self.device.type != dataloader.dataset.device:
                # Prepare for target model device
                batch_inputs = [x.to(self.device) for x in batch_inputs]
                batch_outputs = [y.to(self.device) for y in batch_outputs]

            # Dynamic mask strategy: only applies during training mode
            if num_target_vars == 2 and mode in ['train']:
                if forcast_mask is not None:    # forecasting task
                    batch_inputs[1] = forcast_mask.generate(batch_inputs[1])
                elif impute_mask is not None:   # imputation task
                    intersection_mask = impute_mask.generate(batch_inputs[1])
                    batch_inputs[1] = intersection_mask
                    # This should guarantee that inputs and outputs are consistent.
                    batch_outputs[1] = batch_outputs[1] & (~intersection_mask)

            batch_y_hat = self.model(*batch_inputs)
            batch_loss = self.criterion(batch_y_hat, *batch_outputs)

            if getattr(self.model, 'additional_loss', None) is not None:
                batch_loss += self.model.additional_loss    # KL-Divergence loss or other regularization loss

            if mode in ['train', 'online']:
                self.optimizer.zero_grad()  # clear gradients for next train
                batch_loss.backward()
                self.optimizer.step()

            self.criterion.update(batch_y_hat, *batch_outputs)

            if self.scaler is not None:
                if self.device.type != dataloader.dataset.device:
                    # Prepare for dataset device
                    batch_outputs = [y.to(dataloader.dataset.device) for y in batch_outputs]
                    batch_y_hat = batch_y_hat.to(dataloader.dataset.device)

                batch_y_hat = self.scaler.inverse_transform(batch_y_hat)  # NOTE
                batch_outputs[0] = self.scaler.inverse_transform(*batch_outputs[:num_target_vars])

            self.evaluator.update(batch_y_hat, *batch_outputs)

            if progress_status is not None:
                pbar.set_postfix(batch_loss='{:.6f}'.format(batch_loss.detach().item()))
                pbar.update(1)

        loss = self.criterion.compute()
        metrics = self.evaluator.compute()

        if progress_status is not None:
            pbar.clear()

        return {'loss': loss, **metrics}

    def message_header(self, has_val_dataset: bool = False):
        """
            :return: message header.
        """
        metric_names = ['loss', *self.evaluator.metrics.keys()]
        header = ['train_{}'.format(k) for k in metric_names]
        if has_val_dataset:
            header += ['val_{}'.format(k) for k in metric_names]
        header = ['lr'] + header

        return 'epoch\t' + '\t'.join(['{:^10}'.format(s) for s in header])

    def fit(self, train_dataset: data.Dataset,
            val_dataset: data.Dataset = None,
            epoch_range: Tuple[int, int] = (1, 10),
            batch_size: int = 32, shuffle: bool = False,
            checkpoint_interval: int = 0,
            collate_fn: Callable = None,
            forecast_mask: AbstractMask = None,
            impute_mask: AbstractMask = None,
            verbose: Literal[0, 1, 2] = 2) -> List[List[Union[str, float]]]:

        """
            ``verbose``: 0 is silent. Mostly used for model training only.
                        1 is epoch level, including loss and metrics. Mostly used for command lines to collect outputs;
                        2 is batch level in an epoch, including time. Mostly used for development.
         """

        train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn)
        logger = logging.getLogger()

        message_header = self.message_header(val_dataset is not None)
        if verbose > 0:
            logger.info(message_header)

        performance_history_list = [message_header.split('\t')]
        for epoch in range(epoch_range[0], epoch_range[1] + 1):
            epoch_status = '{}/{}'.format(epoch, epoch_range[1])
            message = [epoch_status, self.optimizer.param_groups[0]['lr']]

            progress_status = ('training ' + epoch_status) if verbose == 2 else None
            train_results = self.run_epoch(train_dataloader, 'train', progress_status, forecast_mask, impute_mask)
            message.extend([*train_results.values()])

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if checkpoint_interval is not None and checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                }, os.path.expanduser('~') + '/pretrained_checkpoint/model_{}.check'.format(epoch))

            if val_dataset is not None:
                progress_status = ('validating ' + epoch_status) if verbose == 2 else None
                val_dataloader = data.DataLoader(val_dataset, batch_size, collate_fn=collate_fn)
                val_results = self.run_epoch(val_dataloader, 'val', progress_status)
                message.extend([*val_results.values()])

                if self.stopper is not None:
                    self.stopper(val_results['loss'])
                    if self.stopper.stop:
                        logger.info('Early stopping at epoch {}, best loss {:.6f}'.format(epoch, self.stopper.best_score))
                        break

            performance_history_list.append(message)

            if verbose > 0:
                logger.info(to_string(*message))

        return performance_history_list

    def evaluate(self, val_dataset: data.Dataset, batch_size: int = 32, collate_fn=None,
                 show_progress: bool = True, is_online: bool = False) -> Dict[str, float]:
        """
            Evaluate the model using ``val_dataset``. Design for evaluation, not for common prediction.

            If ``is_online`` is ``True``, the model will be evaluated in online mode, i.e., one sample at a time.

            :param val_dataset: the data loader of the prediction data.
            :param batch_size: the batch size of a mini-batch.
            :param collate_fn: the function to collate the data.
            :param show_progress: the tqdm desc. If ``None``, not display.
            :param is_online: whether to use online evaluation.
            :return: loss value and metric values.
        """
        if is_online:
            val_mode: Literal['online', 'val'] = 'online'
            val_dataloader = data.DataLoader(val_dataset, 1, collate_fn=collate_fn)
        else:
            val_mode: Literal['online', 'val'] = 'val'
            val_dataloader = data.DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

        results = self.run_epoch(val_dataloader, val_mode, val_mode if show_progress else None)

        return results

    def __str__(self) -> str:
        """
            Print the information of this class instance.
        """
        params = {
            'device': self.device,
            'initial_weights': self.is_initial_weights,
            'compile': self.is_compile,
            'optimizer': type(self.optimizer).__name__,
            'lr': self.optimizer.param_groups[0]['lr'],
            'criterion': type(self.criterion).__name__
        }

        if len(self.evaluator.metrics) > 0:
            params['evaluator'] = '[{}]'.format(', '.join(self.evaluator.metrics.keys()))

        if self.scaler is not None:
            params['scaler'] = type(self.scaler).__name__

        if self.ex_scaler is not None:
            params['ex_scaler'] = type(self.ex_scaler).__name__

        if self.stopper is not None:
            params['stopper'] = str(self.stopper)

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'Trainer({})'.format(params_str)

        return params_str
