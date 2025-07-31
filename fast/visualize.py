#!/usr/bin/env python
# encoding: utf-8

import numpy as np

from typing import Literal
from matplotlib import pyplot as plt

plt.rcParams['figure.max_open_warning'] = 300

# from matplotlib import rcParams
# import matplotlib.font_manager as fm
#
# font_path = '/System/Library/Fonts/Hiragino Sans GB.ttc'
# font_prop = fm.FontProperties(fname=font_path)
# rcParams['font.family'] = font_prop.get_name()


def plot_in_line_chart(time_series: np.ndarray, ts_names: list[str] = None,
                       x_label: str = None, y_label: str = None, title: str = None):
    """
        Plot several **equal-length** time series data in **one** line chart.
        :param time_series: time series data. Shape is ``(n_samples, n_features)``. S.t., n_features >= 1
        :param ts_names: names of each time series.
        :param x_label: x-axis label (string).
        :param y_label: y-axis label (string).
        :param title: title (string) of the plot.
    """
    assert time_series.ndim == 2, 'The time series data should be 2D.'

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams.update({'font.size': 16})

    n_features = time_series.shape[1]
    plt.figure(figsize=(12, 6))
    for i in range(n_features):
        plt.plot(time_series[:, i], label=ts_names[i] if ts_names is not None else f'ts_{i}')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.legend()
    plt.show()


def plot_jagged_ts_in_line_chart(uts: tuple[np.array] or list[np.ndarray], ts_names: list[str] = None,
                                 uts2: tuple[np.array] or list[np.ndarray] = None, ts2_names: list[str] = None,
                                 title: str = None):
    """
        Plot several **unequal-length** (a.k.a. jagged) time series data in one figure of several line charts.
        :param uts: time series list. Each time series shape is ``(n_samples,)``.
        :param ts_names: names of each time series.
        :param uts2: time series list. Each time series shape is ``(n_samples,)``.
        :param ts2_names: names of each time series.
        :param title: title (string) of the plot.
    """
    n_vars = len(uts)
    if ts_names is not None:
        assert len(ts_names) == n_vars, 'The length of ts_names should be equal to the number of uts.'

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams.update({'font.size': 16})

    fig, axs = plt.subplots(n_vars, 1, figsize=(10, 4 * n_vars), squeeze=False)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(title if title is not None else None)

    for i in range(n_vars):
        ax = axs[i, 0]
        ts_data = uts[i]    # shape is (n_samples, 1)
        x = np.arange(len(ts_data))

        ax.plot(x, ts_data, '--o', color='b', label=ts_names[i] if ts_names is not None else None)

        if uts2 is not None:
            ts2_data = uts2[i]
            x = np.arange(len(ts2_data))
            ax.plot(x, ts2_data, '-d', color='r', label=ts2_names[i] if ts2_names is not None else None)

        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_comparable_line_charts(real_ts: np.ndarray, preds_ts: np.ndarray,
                                title: str = None, x_label_names: list[str] = None, y_label_names: list[str] = None):
    """
        Plot several sub-figures in a figure.
        Each sub-figure is a line chart of comparable real and predicted time series data.
        :param real_ts: real time series data. Shape is ``(n_samples, n_features)``. S.t., n_features >= 1
        :param preds_ts: predicted time series data. Shape is ``(n_samples, n_features)``. S.t., n_features >= 1
        :param title: figure title.
        :param x_label_names: names of each real time series.
        :param y_label_names: names of each predicted time series.
    """
    assert real_ts.ndim == 2 and preds_ts.ndim == 2, 'The time series data should be 2D.'
    assert real_ts.shape == preds_ts.shape, 'The shape of real and predicted time series should be the same.'

    seq_len, n_features = real_ts.shape
    x = np.arange(seq_len)

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams.update({'font.size': 16})

    fig, axs = plt.subplots(n_features, 1, figsize=(10, 4 * n_features), squeeze=False)
    fig.subplots_adjust(hspace=0.4)
    fig.suptitle(title if title is not None else None)

    for i in range(n_features):
        ax = axs[i, 0]
        ax.plot(x, real_ts[:, i], '--o', color='r', label='real')
        ax.plot(x, preds_ts[:, i], '-d', color='g', label='preds')

        ax.set_xlabel(x_label_names[i] if x_label_names is not None else None)
        ax.set_ylabel(y_label_names[i] if y_label_names is not None else None)

        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
