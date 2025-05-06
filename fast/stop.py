#!/usr/bin/env python
# encoding: utf-8

from typing import Literal, Callable


class EarlyStop:
    """
        Early stopper to stop the training when the loss does not improve after certain epochs.

        :param patience: How long to wait after last time validation loss improved. Default is 3.
        :param delta:  Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        :param mode: 'abs' for absolute change, 'rel' for relative change (in percentage). Default is 'abs'.
        :param verbose: If True, prints a message for each validation loss improvement. Default is False.
    """

    def __init__(self, patience: int = 3, delta: float = 0, mode: Literal['abs', 'rel']= 'abs', verbose: bool = False):

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: float = None
        self.stop: bool = False

        self._get_threshold: Callable[[float], float] = (lambda x: delta if mode == 'abs' else delta * x)

    def __call__(self, loss: float):
        score = loss
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score + self._get_threshold(self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

            if self.verbose:
                print(f'EarlyStop counter: {loss}, {self.counter} out of {self.patience}.')
        else:   # score < self.best_score + self.delta
            self.best_score = score
            self.counter = 0

    def __str__(self):
        """
            :return: The string of the information of this class instance.
        """

        params = {
            'patience': self.patience,
            'delta': self.delta,
            'mode': self.mode,
            'verbose': self.verbose,
            'counter': self.counter,
            'stop': self.stop,
            'best_score': self.best_score,
            'threshold': self._get_threshold(self.best_score),
        }

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'EarlyStop({})'.format(params_str)

        return params_str

