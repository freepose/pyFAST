#!/usr/bin/env python
# encoding: utf-8

from typing import Literal, Callable


class EarlyStop:
    """
        Early stopper to stop the training when the loss does not improve after certain epochs.

        :param patience: How long to wait after last time validation loss improved. Default is 3.
        :param delta:  Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        :param mode: 'abs' for absolute change, 'rel' for relative change (in percentage). Default is 'abs'.
    """

    def __init__(self, patience: int = 3, delta: float = 0, mode: Literal['abs', 'rel'] = 'abs'):

        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score: float = None
        self.current_score: float = None
        self.counter: int = 0
        self.stop: bool = False

        self._get_threshold: Callable[[float], float] = (lambda x: delta if self.mode == 'abs' else delta * x)

    def __call__(self, loss: float):
        self.current_score = loss
        if self.best_score is None:
            self.best_score = loss

        elif loss >= self.best_score + self._get_threshold(self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        elif loss < self.best_score:
            self.best_score = loss
            self.counter = 0

    def __str__(self):
        """
            String representation of the EarlyStop instance.
        """
        if self.best_score is None:
            params = {
                'patience': self.patience,
                'delta': self.delta,
                'mode': self.mode,
                'counter': self.counter,
                'stop': self.stop,
            }
        else:
            change = self.current_score - self.best_score
            if self.mode == 'rel':
                change /= self.best_score

            params = {
                'change': f"{change * 100:.2f}%" if self.mode == 'rel' else f"{change:.6f}",
                'threshold': f"{self._get_threshold(self.best_score):.6f}",
                'counter': self.counter,
                'stop': self.stop,
            }

        params_str = ', '.join([f'{key}={value}' for key, value in params.items()])
        params_str = 'EarlyStop({})'.format(params_str)

        return params_str
