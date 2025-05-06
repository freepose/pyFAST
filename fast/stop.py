#!/usr/bin/env python
# encoding: utf-8


class EarlyStop:
    """
        Early stopper to stop the training when the loss does not improve after certain epochs.

        :param patience: How long to wait after last time validation loss improved. Default is 3.
        :param delta:  Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        :param verbose: If True, prints a message for each validation loss improvement. Default is False.
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
        else:   # score < self.best_score + self.delta
            self.best_score = score
            self.counter = 0
