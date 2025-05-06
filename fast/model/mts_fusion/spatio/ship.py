#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from ...mts import GAR, AR, VAR

import torch
import torch.nn as nn


class ShipEx(nn.Module):
    def __init__(self, input_window_size: int, input_vars: int = 2, output_window_size: int = 1,
                 ex_retain_window_size: int = 1, ex_vars: int = 3):
        super(ShipEx, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.ex_retain_window_size = ex_retain_window_size
        self.ex_vars = ex_vars

        # 轨迹预测模块：输出(x, y)
        self.traj_predictor = GAR(
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size * 2  # 每个时间步输出2个值
        )

        # 物理参数预测模块（预测速度变化Δu/Δv和转向速率r）
        self.physical_predictor = nn.LSTM(
            input_size=self.ex_vars,
            hidden_size=16,
            batch_first=True
        )
        self.physical_fc = nn.Linear(16, 3)  # 输出(Δu, Δv, r_pred)

        # 可学习的损失权重参数
        self.lambda_kinematic = nn.Parameter(torch.tensor(0.1))
        self.lambda_dynamics = nn.Parameter(torch.tensor(0.01))
        self.lambda_lateral = nn.Parameter(torch.tensor(0.1))

        # 存储附加损失
        self.additive_loss = None

    def forward(self, x: torch.Tensor, ex: torch.Tensor):

        # --- 轨迹预测 ---
        # 输出形状：[batch, output_window, 2]
        traj_output = self.traj_predictor(x)

        # --- 物理参数预测 ---
        # 预测速度变化量和转向速率
        _, (h, _) = self.physical_predictor(ex)
        physical_params = self.physical_fc(h.squeeze(0))  # [batch, 3]
        delta_u = physical_params[:, 0]
        delta_v = physical_params[:, 1]
        r_pred = physical_params[:, 2]

        # --- 提取历史物理参数 ---
        psi0 = ex[:, -1, 0]  # 初始艏向角 [batch]
        u0 = ex[:, -1, 1]  # 初始纵向速度 [batch]
        v0 = ex[:, -1, 2]  # 初始横向速度 [batch]

        # --- 运动学一致性损失计算 ---
        # 计算实际位移（坐标差）
        dx_real = traj_output[:, 1:, 0] - traj_output[:, :-1, 0]  # [batch, pred_len-1]
        dy_real = traj_output[:, 1:, 1] - traj_output[:, :-1, 1]

        # 理论位移（基于物理参数）
        dt = 1.0  # 假设时间间隔为1单位
        u = u0.unsqueeze(-1) + delta_u.unsqueeze(-1) * torch.arange(
            self.output_window_size, device=x.device).float() / self.output_window_size
        v = v0.unsqueeze(-1) + delta_v.unsqueeze(-1) * torch.arange(
            self.output_window_size, device=x.device).float() / self.output_window_size
        psi = psi0.unsqueeze(-1) + r_pred.unsqueeze(-1) * torch.arange(
            self.output_window_size, device=x.device).float() * dt

        # dx_theory = u[:, :-1] * torch.cos(psi[:, :-1]) - v[:, :-1] * torch.sin(psi[:, :-1])
        # dy_theory = u[:, :-1] * torch.sin(psi[:, :-1]) + v[:, :-1] * torch.cos(psi[:, :-1])

        dx_theory = u * torch.cos(psi) - v * torch.sin(psi)
        dy_theory = u * torch.sin(psi) + v * torch.cos(psi)

        kinematic_loss = torch.mean((dx_real - dx_theory) ** 2 + (dy_real - dy_theory) ** 2)

        # --- 动力学平滑性损失 ---
        dynamics_loss = torch.mean(delta_u ** 2 + delta_v ** 2 + r_pred ** 2)

        # --- 横向速度约束 ---
        lateral_loss = torch.mean(v ** 2)  # 约束所有预测时间步的v

        # --- 总附加损失 ---
        self.additive_loss = (
                # self.lambda_kinematic * kinematic_loss +
                self.lambda_dynamics * dynamics_loss +
                self.lambda_lateral * lateral_loss
        )

        return traj_output[:, ::2,:]  # 形状 [batch, pred_len, 2]
