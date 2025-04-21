#!/usr/bin/env python
# encoding: utf-8
import torch


def mean_absolute_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        MAE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: MAE
    """
    return (real - prediction).abs().mean()


def mean_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        MAPE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: MAPE
    """
    errors = real - prediction
    mape = (errors / real).abs().mean()

    return mape


def median_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        MdAPE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: MdAPE
    """
    errors = real - prediction
    # mdape = (errors / real).abs().median(dim=0).values.mean()
    mdape = (errors / real).abs().median()

    return mdape


def standard_deviation_relative_errors(real: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    """
        Standard Deviation of Relative Errors (SDRE).
        "It is used to measure the fluctuation of errors between predicted values and actual values.
        By calculating the relative error of all forecast errors and taking the standard deviation of these errors,
        one can understand how widely the errors are distributed.
        This metric is very useful for evaluating the stability and consistency of a forecasting model."
        :param real:        Real values (1d, 2d, or 3d torch array)
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: SDRE.
    """

    # np.std(np.abs((y_true - y_pred) / y_true))
    errors = real - prediction
    relative_errors = errors / real
    sdre = torch.std(relative_errors)

    return sdre


def cutoff_mean_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor,
                                          cutoff: float = 25000) -> torch.Tensor:
    """
        cutMAPE. Element-wise metrics. A special case for pinn-wpf.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :param cutoff:  Precision cutoff values.
        :return: MAPE
    """
    prediction = prediction[torch.abs(real) > cutoff]
    real = real[torch.abs(real) > cutoff]

    errors = real - prediction
    mape = (errors / real).abs().mean()

    return mape


def symmetric_mean_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        sMAPE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: sMAPE
    """
    errors = real - prediction
    ape = (errors / (real + prediction)).abs()
    smape = ape.mean() * 2

    return smape


def symmetric_median_absolute_percentage_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        sMdAPE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: sMdAPE
    """
    errors = real - prediction
    ape = (errors / (real + prediction)).abs()
    # smdape = ape.median(dim=0).values.mean() * 2
    smdape = ape.median() * 2

    return smdape


def mean_squared_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        MSE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: MSE
    """
    return ((real - prediction) ** 2).mean()


def root_mean_squared_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        RMSE, Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: RMSE
    """
    return ((real - prediction) ** 2).mean().sqrt()


def coefficient_of_variation_of_RMSE(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        CV-RMSE. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: coefficient of variation of RMSE
    """
    rmse = torch.sqrt(torch.mean((real - prediction) ** 2))
    return rmse / real.mean()


def r_square(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        R^2 value. Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: $R^2$
    """
    errors = real - prediction
    squared_errors = errors ** 2

    real_mean_difference = real - real.mean()
    squared_real_mean_difference = real_mean_difference ** 2
    ssr = squared_errors.sum()  # sum of residual errors
    sse = squared_real_mean_difference.sum()  # sum of squared errors
    r2 = - ssr / sse + 1.0

    return r2


def relative_absolute_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        Relative absolute error (RAE). Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: CORR
    """
    errors = real - prediction
    real_mean_difference = real - real.mean()
    rae = torch.sqrt(errors.abs().sum() / real_mean_difference.abs().sum())

    return rae


def relative_squared_error(prediction: torch.tensor, real: torch.tensor) -> torch.Tensor:
    """
        Relative squared errors (RSE). Element-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :return: CORR
    """
    squared_errors = (real - prediction) ** 2
    squared_real_mean_difference = (real - real.mean()) ** 2
    rse = torch.sqrt(squared_errors.sum() / squared_real_mean_difference.sum())

    return rse


def empirical_correlation_coefficient(prediction: torch.tensor, real: torch.tensor, bias: float = 0.01) -> torch.Tensor:
    """
        Mean PCC value of MTS PCC(r, p) values. feature-wise metrics.
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :param bias:        Bias value to avoid zero values.
        :return: CORR
    """
    real_mean_difference = real - real.mean(dim=0)
    squared_real_mean_difference = real_mean_difference ** 2

    prediction_mean_difference = prediction - prediction.mean(dim=0)
    squared_prediction_mean_difference = prediction_mean_difference ** 2

    pcc_numerator = (real_mean_difference * prediction_mean_difference).sum(dim=0)
    pcc_denominator = (squared_real_mean_difference.sum(dim=0) * squared_prediction_mean_difference.sum(dim=0)).sqrt()

    # To avoid the pcc_denominator has zero values, we add a small bias, and its numerator is set to 0.
    pcc_denominator[pcc_denominator == 0] += bias
    pcc_numerator[pcc_denominator == 0] = 0.

    pcc = (pcc_numerator / pcc_denominator).mean()

    return pcc


def root_mean_square_percentage_error(prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
        Root Mean Square Percentage Error (RMSPE).
        Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array), note that the real values should be non-negative.
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: RMSPE.
    """
    errors = real - prediction
    rmspe = torch.sqrt(torch.mean((errors / real) ** 2))

    return rmspe


def root_median_square_percentage_error(prediction: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """
        Root Median Square Percentage Error (RMdSPE).
        Element-wise metrics.
        :param real:        Real values (1d, 2d, or 3d torch array), note that the real values should be non-negative.
        :param prediction:  Predicted values (1d, 2d, or 3d torch array)
        :return: RMSPE.
    """
    errors = real - prediction
    rdmspe = torch.sqrt(torch.median((errors / real) ** 2))

    return rdmspe


def mean_absolute_scaled_error(prediction: torch.Tensor, real: torch.Tensor, seasonality: int = 1) -> torch.Tensor:
    """
        Mean Absolute Scaled Error (MASE).
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :param seasonality: seasonality of the data (default 1 for non-seasonal data)
        :return: MASE value.
    """
    assert seasonality > 0, 'The seasonality should be greater than 0.'

    if real.ndim == 1:  # (seq_len,)
        real = real.unsqueeze(1)  # -> (seq_len, 1)
        prediction = prediction.unsqueeze(1)  # -> (seq_len, 1)
    elif real.ndim == 3:  # (batch, seq_len, features)
        real = real.flatten(0, 1)  # -> (seq_len, features)
        prediction = prediction.flatten(0, 1)  # -> (seq_len, features)

    model_errors = real - prediction  # -> (seq_len, features)
    naive_errors = real[seasonality:] - real[:-seasonality]  # -> (seq_len - seasonality, features)

    mae_model = model_errors.abs().mean(dim=0)  # -> (features,)
    mae_naive = naive_errors.abs().mean(dim=0)  # -> (features,)

    return (mae_model / mae_naive).mean()


def median_absolute_scaled_error(prediction: torch.Tensor, real: torch.Tensor, seasonality: int = 1) -> torch.Tensor:
    """
        Median Absolute Scaled Error (MdASE).
        :param prediction:  predicted values (1d, 2d, or 3d torch tensor)
        :param real:        real values (1d, 2d, or 3d torch tensor)
        :param seasonality: seasonality of the data (default 1 for non-seasonal data)
        :return: MASE value.
    """
    assert seasonality > 0, 'The seasonality should be greater than 0.'

    if real.ndim == 1:  # (seq_len,)
        real = real.unsqueeze(1)  # -> (seq_len, 1)
        prediction = prediction.unsqueeze(1)  # -> (seq_len, 1)
    elif real.ndim == 3:  # (batch, seq_len, features)
        real = real.flatten(0, 1)  # -> (seq_len, features)
        prediction = prediction.flatten(0, 1)  # -> (seq_len, features)

    model_errors = real - prediction  # -> (seq_len, features)
    naive_errors = real[seasonality:] - real[:-seasonality]  # -> (seq_len - seasonality, features)

    mae_model = model_errors.abs().median(dim=0).values  # -> (features,)
    mae_naive = naive_errors.abs().median(dim=0).values  # -> (features,)

    return (mae_model / mae_naive).mean()
