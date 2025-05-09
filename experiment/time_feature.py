#!/usr/bin/env python
# encoding: utf-8

"""
    The package is implemented based on pandas (i.e., datetime).
"""

import numpy as np
import pandas as pd

from typing import Literal
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


def second_of_minute(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The second of minute feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the second of minute feature.
    """
    series = index.second
    if normalize:
        series = series / 59.0 - 0.5

    return series


def minute_of_hour(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The minute of hour feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the minute of hour feature.
    """
    series = index.minute
    if normalize:
        series = series / 59.0 - 0.5

    return series


def hour_of_day(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The hour of day feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the hour of day feature.
    """
    series = index.hour
    if normalize:
        series = series / 23.0 - 0.5

    return series


def day_of_week(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The day of week feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the day of week feature.
    """
    series = index.dayofweek
    if normalize:
        series = series / 6.0 - 0.5

    return series


def day_of_month(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The day of month feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the day of month feature.
    """
    series = index.day
    if normalize:
        series = (series - 1) / 30.0 - 0.5

    return series


def day_of_year(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The day of year feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the day of year feature.
    """
    series = index.dayofyear
    if normalize:
        series = (series - 1) / 365.0 - 0.5

    return series


def month_of_year(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The month of year feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the month of year feature.
    """
    series = index.month
    if normalize:
        series = (series - 1) / 11.0 - 0.5

    return series


def week_of_year(index: pd.Series, normalize: bool = False) -> pd.Series:
    """
        The week of year feature is extracted from the datetime index.
        :param index: datetime index.
        :param normalize: whether to normalize the time features into [-.5, 0.5].
        :return: the week of year feature.
    """
    series = index.isocalendar().week
    if normalize:
        series = (index.isocalendar().week - 1) / 52.0 - 0.5

    return series


class TimeAsFeature:
    """
        The features are extracted from datetime.
        Deconstruct datetime into granular components to model linear/categorical patterns.

        The following frequencies are supported:
            Y   - yearly
                alias: A
            ME   - monthly
            W   - weekly
            d   - daily
            B   - business days
            h   - hourly
            min   - minutely
                alias: min
            s   - secondly

        :param freq: time frequency, also known as time granularity/interval.
        :param is_normalized: whether to normalize the time features into [-.5, 0.5].
    """

    def __init__(self, freq: Literal['Y', 'ME', 'W', 'd', 'B', 'h', 'min', 's'] = 'D',
                 is_normalized: bool = True):
        self.freq = freq
        self.normalize = is_normalized

        offset_func_dict = {
            offsets.YearEnd: [],
            offsets.QuarterEnd: [month_of_year],
            offsets.MonthEnd: [month_of_year],
            offsets.Week: [day_of_month, week_of_year],
            offsets.Day: [day_of_week, day_of_month, day_of_year],
            offsets.BusinessDay: [day_of_week, day_of_month, day_of_year],
            offsets.Hour: [hour_of_day, day_of_week, day_of_month, day_of_year],
            offsets.Minute: [minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
            offsets.Second: [second_of_minute, minute_of_hour, hour_of_day, day_of_week, day_of_month, day_of_year],
        }

        offset = to_offset(freq)

        for offset_type, feature_classes in offset_func_dict.items():
            if isinstance(offset, offset_type):
                self.feature_func = feature_classes

    def __call__(self, dates: pd.Series) -> np.ndarray:
        """
            :param dates: input datetime index.
            :return: the features extracted from the datetime index.
        """

        feature_list = []
        for func in self.feature_func:
            _series = func(dates, self.normalize)
            _array = np.array(_series).reshape(-1, 1)
            feature_list.append(_array)

        feature_array = np.concatenate(feature_list, axis=1)
        return feature_array