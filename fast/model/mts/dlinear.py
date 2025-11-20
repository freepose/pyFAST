#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from typing import Literal

from setuptools.monkey import patch_func

from ..base.decomposition import DecomposeSeries
from ...data import PatchMaker
from .ar import GAR, AR


class DLinear(nn.Module):
    """
        **Decomposition-Linear**

        Are Transformers Effective for Time Series Forecasting?
        Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu.
        AAAI 2022, DOI: 10.1609/aaai.v37i9.26317.
        url: https://arxiv.org/pdf/2205.13504.pdf

        Author provided code: https://github.com/cure-lab/LTSF-Linear

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param kernel_size: the kernel size of series decomposition function.
        :param mapping: the mapping type, 'gar' for Global AR, 'ar' for Autoregressive.
        :param enable_ps_loss: whether to enable patch-wise structural loss.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 kernel_size: int = 25, mapping: Literal['gar', 'ar'] = 'gar', enable_ps_loss: bool = False):
        super(DLinear, self).__init__()
        assert mapping in ['gar', 'ar'], f"Mapping should be 'gar' or 'ar', but got {mapping}."
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.kernel_size = kernel_size
        self.mapping = mapping
        self.enable_ps_loss = enable_ps_loss

        if self.enable_ps_loss:
            self.ps_lambda = 3.0
            self.patch_len_threshold = 24
            self.kl_loss = nn.KLDivLoss(reduction='none')

        self.decomposition = DecomposeSeries(kernel_size)

        if mapping == 'ar':
            self.trend_l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
            self.seasonal_l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
        else:
            self.trend_l1 = GAR(self.input_window_size, self.output_window_size)
            self.seasonal_l1 = GAR(self.input_window_size, self.output_window_size)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
            :param x_mask: shape is (batch_size, input_window_size, input_vars), mask tensor.
        """
        if x_mask is not None:
            x[~x_mask] = 0.

        trend_init, seasonal_init = self.decomposition(x)

        trend_output = self.trend_l1(trend_init)
        seasonal_output = self.seasonal_l1(seasonal_init)

        out = seasonal_output + trend_output

        return out

    def fouriour_based_adaptive_patching(self, real, prediction):
        # Get patch length a stride
        true_fft = torch.fft.rfft(real, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        frequency_list[:1] = 0.0
        top_index = torch.argmax(frequency_list)
        period = (real.shape[1] // top_index)
        patch_len = min(period // 2, self.patch_len_threshold)
        stride = patch_len // 2
        patch_maker = PatchMaker(seq_len=self.output_window_size, patch_len=patch_len, patch_stride=stride)
        # Create patch maker
        true_patch = patch_maker(real)
        pred_patch = patch_maker(prediction)

        return true_patch, pred_patch

    def gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
        """
            Gradient based dynamic weighting.
        """
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var)
        pred_std = torch.sqrt(pred_var)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + 1e-5) / (true_std * pred_std + 1e-5)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + 1e-5) / (true_var + pred_var + 1e-5)

        # Gradiant based dynamic weighting
        params = list(self.trend_l1.parameters())
        corr_gradient = torch.autograd.grad(corr_loss, params, create_graph=True)[0]
        var_gradient = torch.autograd.grad(var_loss, params, create_graph=True)[0]
        mean_gradient = torch.autograd.grad(mean_loss, params, create_graph=True)[0]
        gradiant_avg = (corr_gradient + var_gradient + mean_gradient) / 3.0

        alpha = gradiant_avg.norm().detach() / corr_gradient.norm().detach()
        beta = gradiant_avg.norm().detach() / var_gradient.norm().detach()
        gamma = gradiant_avg.norm().detach() / mean_gradient.norm().detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()

        return alpha, beta, gamma

    def patch_wise_structural_loss(self, true_patch, pred_patch):

        # Calculate mean
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)

        # Calculate variance and standard deviation
        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var)
        pred_patch_std = torch.sqrt(pred_patch_var)

        # Calculate Covariance
        true_pred_patch_cov = torch.mean((true_patch - true_patch_mean) * (pred_patch - pred_patch_mean), dim=-1,
                                         keepdim=True)

        # 1. Calculate linear correlation loss
        patch_linear_corr = (true_pred_patch_cov + 1e-5) / (true_patch_std * pred_patch_std + 1e-5)
        linear_corr_loss = (1.0 - patch_linear_corr).mean()

        # 2. Calculate variance
        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self.kl_loss(pred_patch_softmax, true_patch_softmax).sum(dim=-1).mean()

        # 3. Mean loss
        mean_loss = torch.abs(true_patch_mean - pred_patch_mean).mean()

        return linear_corr_loss, var_loss, mean_loss

    def output_aware_loss(self, prediction: torch.Tensor, real: torch.Tensor, real_mask: torch.Tensor) -> torch.Tensor:
        """
            Patch-wise Structural loss. Currently, not support for multi-gpu training and mask modeling.
        """
        # Fourier based adaptive patching
        if self.enable_ps_loss:
            true_patch, pred_patch = self.fouriour_based_adaptive_patching(real, prediction)

            # Patch-wise structural loss
            corr_loss, var_loss, mean_loss = self.patch_wise_structural_loss(true_patch, pred_patch)

            # Gradient based dynamic weighting
            alpha, beta, gamma = self.gradient_based_dynamic_weighting(real, prediction, corr_loss, var_loss, mean_loss)

            # Final PS loss
            ps_loss = alpha * corr_loss + beta * var_loss + gamma * mean_loss

            ps_loss = ps_loss * self.ps_lambda
        else:
            ps_loss = 0.

        return ps_loss


class NLinear(nn.Module):
    """
        **Normalization-Linear**

        Are Transformers Effective for Time Series Forecasting?
        Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu
        url: https://arxiv.org/pdf/2205.13504.pdf

        Author provided code: https://github.com/cure-lab/LTSF-Linear

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param mapping: the mapping type, 'gar' for Global AR, 'ar' for Autoregressive.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 mapping: Literal['gar', 'ar'] = 'gar'):
        super(NLinear, self).__init__()
        assert mapping in ['gar', 'ar'], f"Mapping should be 'gar' or 'ar', but got {mapping}."
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.mapping = mapping

        if self.mapping == 'ar':
            self.l1 = AR(self.input_window_size, self.input_vars, self.output_window_size)
        else:
            self.l1 = GAR(self.input_window_size, self.output_window_size)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
            :param x_mask: shape is (batch_size, input_window_size, input_vars), mask tensor.
        """
        if x_mask is not None:
            x[~x_mask] = 0.

        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        x = self.l1(x)    # -> (batch_size, output_window_size, input_vars)

        out = x + seq_last

        return out
