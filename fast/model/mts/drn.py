#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """
        Because the original Block module cannot do multivariable output without using rnn,
        we changed the output sizes of ``self.backcast_out`` and ``self.forecast_out``
        by multiplying them by ``output_vars``,
        and unflatten the result of the calculation before the forward function outputs the final result.
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int, output_vars: int,
                 hidden_size: int, use_rnn: bool):
        super(Block, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.use_rnn = use_rnn

        if use_rnn:
            self.lstm = nn.LSTM(input_vars, hidden_size, num_layers=2, batch_first=True, bidirectional=True)
            self.lin = nn.Linear(hidden_size * 2, (input_window_size + output_window_size) * output_vars)
        else:
            self.lin1 = nn.Linear(input_window_size * input_vars, hidden_size)
            self.lin2 = nn.Linear(hidden_size, hidden_size)
            self.lin3 = nn.Linear(hidden_size, hidden_size)
            self.lin4 = nn.Linear(hidden_size, hidden_size)
            self.backcast_layer = nn.Linear(hidden_size, hidden_size)
            self.forecast_layer = nn.Linear(hidden_size, hidden_size)
            self.backcast_out = nn.Linear(hidden_size, input_window_size * output_vars)
            self.forecast_out = nn.Linear(hidden_size, output_window_size * output_vars)

    def forward(self, x: torch.Tensor):
        """
        :param x: shape is (batch_size, input_window_size, input_vars).
        """

        if self.use_rnn:
            lstm_out, _ = self.lstm(x)
            out = self.lin(lstm_out[:, -1, :])
            backcast = out[:, :self.input_window_size * self.output_vars]
            forecast = out[:, self.input_window_size * self.output_vars:]
        else:
            x = x.flatten(1, -1)  # -> (batch_size, input_window_size * input_vars)
            x = F.relu(self.lin1(x))  # -> (batch_size, hidden_size)
            x = F.relu(self.lin2(x))
            x = F.relu(self.lin3(x))
            x = F.relu(self.lin4(x))
            theta_b = F.relu(self.backcast_layer(x))
            theta_f = F.relu(self.forecast_layer(x))
            backcast = self.backcast_out(theta_b)  # -> (batch_size, input_window_size)
            forecast = self.forecast_out(theta_f)  # -> (batch_size, output_window_size)

        backcast = backcast.unflatten(-1, (self.input_window_size, self.output_vars))  # add unflatten
        forecast = forecast.unflatten(-1, (self.output_window_size, self.output_vars))  # add unflatten
        return backcast, forecast


class Stack(nn.Module):
    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int, output_vars: int,
                 number_blocks_per_stack: int, hidden_size: int, use_rnn: bool):
        super(Stack, self).__init__()
        self.blocks = nn.ModuleList(
            [Block(input_window_size, input_vars, output_window_size, output_vars, hidden_size, use_rnn)
             for _ in range(number_blocks_per_stack)])

    def forward(self, forecast: torch.Tensor, backcast: torch.Tensor, backsum: torch.Tensor):
        for block in self.blocks:
            b, f = block(backcast)

            # original codes
            # backcast2 = backcast.clone()
            # backcast2[:, :, 0] = backcast2[:, :, 0] - b
            # backcast2 = backcast2 - b
            # backcast = backcast2
            # backsum = backsum + b

            backcast = backcast.clone() - b
            backsum = backsum + b

            forecast = forecast + f

        return forecast, backcast, backsum


class DeepResidualNetwork(nn.Module):
    """
        Harry Rubin-Falcone, Ian Fox, and Jenna Wiens
        Deep Residual Time-Series Forecasting: Application to Blood Glucose Prediction, KDH@ECAI 2020
        url: https://api.semanticscholar.org/CorpusID:221803676

        The article does not give the model a precise name, we will name it DeepResidualNetwork based on its structure.

        Official Code: https://github.com/MLD3/Deep-Residual-Time-Series-Forecasting

        :param input_window_size: input window size.
        :param input_vars: input variable number.
        :param output_window_size: output window size.
        :param output_vars: output size.
        :param hidden_size: hidden size of ``Block``.
        :param number_stacks: the number of stack layers。
        :param number_blocks_per_stack: the number of blocks per stack layers.
        :param use_rnn: whether to use rnn modules inside the block, or use linear modules.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1,
                 output_window_size: int = 1, output_vars: int = 1,
                 hidden_size: int = 32, number_stacks: int = 1, number_blocks_per_stack: int = 1, use_rnn: bool = True):
        super(DeepResidualNetwork, self).__init__()
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.hidden_size = hidden_size
        self.number_stacks = number_stacks
        self.number_blocks_per_stack = number_blocks_per_stack
        self.use_rnn = use_rnn

        self.stacks = nn.ModuleList([
            Stack(input_window_size, input_vars, output_window_size, output_vars,
                  number_blocks_per_stack, hidden_size, use_rnn)
            for _ in range(number_stacks)])

    def forward(self, x: torch.Tensor):
        """
            :param x: shape is (batch_size, input_window_size, input_vars).
        """

        forecast = torch.zeros((x.shape[0], self.output_window_size, self.output_vars),
                               device=x.device)  # add third dimension
        backsum = torch.zeros((x.shape[0], self.input_window_size, self.output_vars),
                              device=x.device)  # add third dimension

        for stack in self.stacks:
            forecast, x, backsum = stack(forecast, x, backsum)

        return forecast
