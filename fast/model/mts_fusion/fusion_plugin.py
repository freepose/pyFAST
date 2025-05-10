#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from fast import get_common_kwargs
from fast.model.mts import GAR


class DataFirstPlugin(nn.Module):
    """
        Data first plugin for fusing target and exogenous variables.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param ex_retain_window_size: exogenous input window size.
        :param ex_vars: number of exogenous input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param model_cls: backbone model class.
        :param params: the hyperparameters of backbone model.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 model_cls: nn.Module = GAR, params: dict = {}):
        super(DataFirstPlugin, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        if input_window_size != ex_retain_window_size:
            self.ex_gar = GAR(ex_retain_window_size, input_window_size)

        ds_dict = {'input_window_size': input_window_size, 'input_vars': input_vars + ex_vars,
                   'output_window_size': output_window_size, 'output_vars': output_vars}

        common_ds_params = get_common_kwargs(model_cls.__init__, ds_dict)
        model_settings = {**common_ds_params, **params}

        self.model_inst = model_cls(**model_settings)
        if 'output_vars' not in common_ds_params:
            self.model_inst = nn.Sequential(self.model_inst, nn.Linear(input_vars + ex_vars, output_vars))

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            x -> [batch_size, input_window_size, input_vars]
            ex -> [batch_size, ex_retain_window_size, ex_vars]
        """
        ex = ex[:, -self.ex_retain_window_size:]

        if self.input_window_size != self.ex_retain_window_size:
            ex = self.ex_gar(ex)  # -> [batch_size, input_window_size, ex_vars]

        xe = torch.cat([x, ex], dim=-1)  # -> [batch_size, input_window_size, input_vars + ex_vars]
        out = self.model_inst(xe)  # -> [batch_size, output_window_size, output_size]

        return out


class LearningFirstPlugin(nn.Module):
    """
        Learning first plugin for fusing target variables and exogenous variables.

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param ex_retain_window_size: exogenous input window size.
        :param ex_vars: number of exogenous input variables.
        :param output_window_size: output window size.
        :param output_vars: number of output variables.
        :param model_cls: backbone model class.
        :param params: the hyperparameters of backbone model.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 model_cls: nn.Module = GAR, params: dict = {}):
        super(LearningFirstPlugin, self).__init__()

        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars

        ds_dict = {'input_window_size': input_window_size, 'input_vars': input_vars,
                   'output_window_size': output_window_size, 'output_vars': input_vars}
        common_ds_params = get_common_kwargs(model_cls.__init__, ds_dict)
        model_settings = {**common_ds_params, **params}
        self.model_inst = model_cls(**model_settings)

        ex_ds_dict = {'input_window_size': input_window_size, 'input_vars': ex_vars,
                      'output_window_size': output_window_size, 'output_vars': ex_vars}
        ex_common_ds_params = get_common_kwargs(model_cls.__init__, ex_ds_dict)
        ex_model_settings = {**ex_common_ds_params, **params}
        self.ex_model_inst = model_cls(**ex_model_settings)

        self.fc = nn.Linear(input_vars + ex_vars, output_vars)

    def forward(self, x: torch.Tensor, ex: torch.Tensor):
        """
            x -> [batch_size, input_window_size, input_vars]
            ex -> [batch_size, ex_retain_window_size, ex_vars]
        """
        ex = ex[:, -self.ex_retain_window_size:]

        x = self.model_inst(x)  # -> [batch_size, output_window_size, input_vars]
        ex = self.ex_model_inst(ex)  # -> [batch_size, output_window_size, ex_vars]

        xe = torch.cat([x, ex], dim=-1)  # -> [batch_size, output_window_size, input_vars + ex_vars]
        out = self.fc(xe)  # -> [batch_size, output_window_size, output_vars]

        return out


class ExogenousDataDrivenPlugin(nn.Module):
    """
        Exogenous Data-Driven Modeling (For pKa prediction/estimation).
        For pKa estimation, assure that ``input_window_size`` == ``output_window_size``.
    """

    def __init__(self, input_window_size, ex_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 ex_model_cls: nn.Module = GAR, params: dict = {}):
        super(ExogenousDataDrivenPlugin, self).__init__()

        ex_ds_dict = {'input_window_size': input_window_size, 'input_vars': ex_vars,
                      'output_window_size': output_window_size, 'output_vars': output_vars}
        ex_ds_params = get_common_kwargs(ex_model_cls.__init__, ex_ds_dict)
        model_settings = {**ex_ds_params, **params}
        self.ex_model_inst = ex_model_cls(**model_settings)

        if 'output_vars' not in ex_ds_params:
            self.ex_model_inst = nn.Sequential(self.ex_model_inst, nn.Linear(ex_vars, output_vars))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, ex: torch.Tensor):
        """
            :param x: shape is (batch_size, input_window_size, input_vars), data type is float.
            :param x_mask: shape is (batch_size, input_window_size, input_vars), data type is bool.
            :param ex: shape is (batch_size, input_window_size, ex_vars), data type is float.
            :return: shape is (batch_size, output_window_size, output_vars), data type is float.
        """

        out = self.ex_model_inst(ex)  # -> (batch_size, output_window_size, output_vars)

        return out
