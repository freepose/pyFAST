#!/usr/bin/env python
# encoding: utf-8

"""
    This package supports training all models, which includes sparse data and mask models in terms of uts and mts.
    Single computer/server version.
"""

import os, sys, platform
import torch
import torch.nn as nn
import torch.utils.data as data

from tqdm import tqdm

from .model.base import to_string, init_weights
from .visualize import plot_comparable_line_charts


class EarlyStop:
    """
        Early stopper to stop the training when the loss does not improve after certain epochs.
        :param patience: How long to wait after last time validation loss improved. Default is 3.
        :param delta:  Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        :param verbose:
    """

    def __init__(self, patience: int = 3, delta: float = 0, verbose: bool = False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: float = None
        self.stop: bool = False

    def __call__(self, loss: float):
        score = loss
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

            if self.verbose:
                print(f'EarlyStop counter: {loss}, {self.counter} out of {self.patience}.')
        else:
            self.best_score = score
            self.counter = 0


class Trainer:
    def __init__(self, device: torch.device,
                 model: nn.Module, is_initial_weights: bool = False, is_compile: bool = False,
                 optimizer=None, lr_scheduler=None, stopper=None,
                 criterion=nn.MSELoss(), additive_criterion=None, evaluator=None,
                 global_scaler=None, global_ex_scaler = None):
        self.device = device

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.stopper = stopper

        self.criterion = criterion
        self.additive_criterion = additive_criterion
        self.evaluator = evaluator

        self.global_scaler = global_scaler
        self.global_ex_scaler = global_ex_scaler

        self.model = model
        self.is_initial_weights = is_initial_weights
        self.is_compile = is_compile

        if self.is_initial_weights:
            model.apply(init_weights)

        self.initialize_device()

        if self.is_compile:
            self.model = torch.compile(self.model)

    def initialize_device(self):
        """ initialize accelerator. """
        self.model = self.model.to(self.device)

        if self.device.type == 'cpu':
            if platform.machine() == 'x86_64':
                torch.set_num_threads(os.cpu_count() - 2)
            elif platform.machine() == 'arm64':
                torch.set_num_threads(os.cpu_count())
        elif self.device.type == 'cuda':
            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
        elif self.device.type == 'mps':
            pass

    def fit(self, train_dataset: data.Dataset, val_dataset: data.Dataset = None,
            epoch_range: tuple[int, int] = (1, 10), verbose: bool = True,
            batch_size: int = 32, shuffle: bool = False,
            checkpoint_interval: int = 0, display_interval: int = 0,
            generation_interval: int = None,
            collate_fn=None) -> list:
        """
            Train the model.
            :param train_dataset: training dataset.
            :param val_dataset: validation dataset.
            :param epoch_range: the range of epochs.
            :param verbose: whether to display the training process.
            :param batch_size: the batch size of a mini-batch.
            :param shuffle: whether to shuffle the data.
            :param checkpoint_interval: the interval of saving model.
            :param display_interval: the interval of visualization of predictions.
            :param generation_interval: the interval of model generation evaluation. Support from generative models.
            :param collate_fn: the function to collate the data.
            :return: the trained performance list.
        """
        train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn)
        train_dataloader2 = data.DataLoader(train_dataset, batch_size, collate_fn=collate_fn)

        performance_list = []
        for epoch in range(epoch_range[0], epoch_range[1] + 1):
            message_prefix = '{}/{}'.format(epoch, epoch_range[1])
            message = [message_prefix, self.optimizer.param_groups[0]['lr']]

            self.model.train()
            self.fit_step(train_dataloader, 'training ' + message_prefix)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if checkpoint_interval is not None and checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "lr_scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                }, 'D:/checkpoint/model_{}.check'.format(epoch))

            if not verbose:
                continue

            y_train_hat, y_train = self.predict(train_dataloader2, 'evaluate train ' + message_prefix)

            train_loss = self.criterion(y_train_hat, *y_train)
            train_metric_dict = self.evaluator.evaluate(y_train_hat, *y_train)
            message.extend([train_loss, *train_metric_dict.values()])

            if self.stopper is not None:
                self.stopper(train_loss)
                if self.stopper.stop:
                    break

            if val_dataset is not None:
                val_dataloader = data.DataLoader(val_dataset, batch_size, collate_fn=collate_fn)

                y_val_hat, y_val = self.predict(val_dataloader, 'evaluate val ' + message_prefix)

                val_loss = self.criterion(y_val_hat, *y_val)
                val_metric_dict = self.evaluator.evaluate(y_val_hat, *y_val)
                message.extend([val_loss, *val_metric_dict.values()])

                if display_interval is not None and display_interval > 0 and epoch % display_interval == 0:
                    plot_num_vars = min(y_val_hat.shape[-1], 4)

                    if len(y_val) == 1:
                        real_tensor = y_val[0].flatten(0, 1).detach()
                        real_array = real_tensor.cpu().numpy()[:, :plot_num_vars]
                        preds_tensor = y_val_hat.flatten(0, 1).detach()
                        preds_array = preds_tensor.cpu().numpy()[:, :plot_num_vars]

                        plot_msg = 'epoch:' + message_prefix + ' '
                        plot_msg += ', '.join(['{}:{:.4f}'.format(k, v) for k, v in val_metric_dict.items()])

                        plot_comparable_line_charts(real_array, preds_array, plot_msg)

                if generation_interval is not None and generation_interval > 0 and epoch % generation_interval == 0:
                    generate_fn = getattr(self.model, 'generate', None)
                    if generate_fn is not None:
                        plot_num_vars = min(y_val_hat.shape[-1], 4)

                        y_gen_hat, y_gen = self.generate(val_dataloader, 'generate val ' + message_prefix)

                        gen_loss = self.criterion(y_gen_hat, *y_gen)
                        gen_metric_dict = self.evaluator.evaluate(y_gen_hat, *y_gen)
                        message.extend([gen_loss, *gen_metric_dict.values()])

                        real_tensor = y_gen[0].flatten(0, 1).detach()
                        real_array = real_tensor.cpu().numpy()[:, :plot_num_vars]
                        preds_tensor = y_gen_hat.flatten(0, 1).detach()
                        preds_array = preds_tensor.cpu().numpy()[:, :plot_num_vars]

                        plot_msg = 'epoch:' + message_prefix + ' '
                        plot_msg += ', '.join(['{}:{:.4f}'.format(k, v) for k, v in gen_metric_dict.items()])

                        plot_comparable_line_charts(real_array, preds_array, plot_msg)

            performance_list.append(message)
            print(to_string(*message))

        return performance_list

    def fit_step(self, dataloader: data.DataLoader, tqdm_desc: str = None):
        """
            A pass of all trainable data.
            :param dataloader: the data loader of the trainable data.
            :param tqdm_desc: the description of tqdm.
            :return: the loss value of this pass.
        """
        with tqdm(total=len(dataloader), leave=False, file=sys.stdout) as pbar:
            pbar.set_description(tqdm_desc)

            for batch_inputs, batch_outputs in dataloader:

                num_x = len(batch_outputs)
                num_ex = len(batch_inputs) - num_x

                if self.global_scaler is not None:
                    batch_inputs[0] = self.global_scaler.transform(*batch_inputs[:num_x])
                    batch_outputs[0] = self.global_scaler.transform(*batch_outputs[:num_x])

                if num_ex > 0 and (self.global_ex_scaler is not None):
                    batch_inputs[num_x] = self.global_ex_scaler.transform(batch_inputs[num_x])

                if self.device.type != dataloader.dataset.device:
                    batch_inputs = [x.to(self.device) for x in batch_inputs]
                    batch_outputs = [y.to(self.device) for y in batch_outputs]

                batch_y_hat = self.model(*batch_inputs)

                batch_loss = self.criterion(batch_y_hat, *batch_outputs)

                if self.additive_criterion is not None:
                    additive_loss = self.additive_criterion(*batch_inputs)
                    batch_loss = batch_loss + additive_loss

                self.optimizer.zero_grad()  # clear gradients for next train
                batch_loss.backward()
                self.optimizer.step()

                pbar.set_postfix(batch_loss='{:.6f}'.format(batch_loss.detach().item()))
                pbar.update(1)

    def predict(self, dataloader: data.DataLoader, tqdm_desc: str = None) -> tuple:
        """
            Predict the model using ``dataloader``. Design for evaluation, not for common prediction.
            :param dataloader: the data loader of the prediction data.
            :param tqdm_desc: the description of tqdm.
            :return: the prediction values and real values.
        """
        self.model.eval()

        with tqdm(total=len(dataloader), leave=False, file=sys.stdout) as pbar:
            pbar.set_description(tqdm_desc)

            y_list, y_hat_list = [], []
            for batch_inputs, batch_outputs in dataloader:

                num_x = len(batch_outputs)
                num_ex = len(batch_inputs) - num_x
                if self.global_scaler is not None:
                    batch_inputs[0] = self.global_scaler.transform(*batch_inputs[:num_x])

                if num_ex > 0 and (self.global_ex_scaler is not None):
                    batch_inputs[num_x] = self.global_ex_scaler.transform(batch_inputs[num_x])

                if self.device.type != dataloader.dataset.device:
                    batch_inputs = [x.to(self.device) for x in batch_inputs]

                batch_y_hat = self.model(*batch_inputs).detach()
                if self.device.type != dataloader.dataset.device:
                    batch_y_hat = batch_y_hat.to(dataloader.dataset.device)

                if self.global_scaler is not None:
                    batch_y_hat = self.global_scaler.inverse_transform(batch_y_hat)

                y_hat_list.append(batch_y_hat)
                y_list.append(batch_outputs)

                pbar.update(1)

            y_hat = torch.cat(y_hat_list, dim=0)
            y = [torch.cat(tensors, dim=0) for tensors in zip(*y_list)]

            return y_hat, y

    def generate(self, dataloader: data.DataLoader, tqdm_desc: str = None) -> tuple:
        """
            Predict the model using ``dataloader``. Design for evaluation, not for common prediction.
            :param dataloader: the data loader of the prediction data.
            :param tqdm_desc: the description of tqdm.
            :return: the prediction values and real values.
        """
        self.model.eval()

        with tqdm(total=len(dataloader), leave=False, file=sys.stdout) as pbar:
            pbar.set_description(tqdm_desc)

            y_list, y_hat_list = [], []
            for batch_inputs, batch_outputs in dataloader:
                num_x = len(batch_outputs)
                num_ex = len(batch_inputs) - num_x

                # if self.global_scaler is not None:
                #     batch_inputs[0] = self.global_scaler.transform(*batch_inputs[:num_x])

                if num_ex > 0 and (self.global_ex_scaler is not None):
                    batch_inputs[num_x] = self.global_ex_scaler.transform(*batch_inputs[num_x:num_x + 1])

                if self.device.type != dataloader.dataset.device:
                    batch_inputs = [x.to(self.device) for x in batch_inputs]

                batch_y_hat = self.model.generate(*batch_inputs[num_x:]).detach()

                if self.device.type != dataloader.dataset.device:
                    batch_y_hat = batch_y_hat.to(dataloader.dataset.device)

                if self.global_scaler is not None:
                    batch_y_hat = self.global_scaler.inverse_transform(batch_y_hat)

                y_hat_list.append(batch_y_hat)
                y_list.append(batch_outputs)

                pbar.update(1)

            y_hat = torch.cat(y_hat_list, dim=0)
            y = [torch.cat(tensors, dim=0) for tensors in zip(*y_list)]

            return y_hat, y
