# !/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class LSTDLongEncoder(nn.Module):
    """
        Long-term state encoder using convolutional layers.
        Produces a long-term latent variable (mean and logvar) from input sequence.
    """

    def __init__(self, input_vars: int, hidden_dim: int, latent_dim: int):
        super(LSTDLongEncoder, self).__init__()
        # 1D convolution to extract long-term features
        self.conv1 = nn.Conv1d(in_channels=input_vars, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        # Global average pooling to collapse time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Fully connected layers to output mean and log-variance of long latent variable
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of shape (batch_size, seq_len, input_vars)
        :return: z_d (sampled latent), mu_d, logvar_d
        """
        # x -> (batch, input_vars, seq_len)
        h = x.permute(0, 2, 1)
        h = self.act(self.conv1(h))
        # (batch, hidden_dim, seq_len) -> (batch, hidden_dim, 1)
        h = self.pool(h)
        h = h.squeeze(-1)  # (batch, hidden_dim)
        mu = self.fc_mu(h)  # (batch, latent_dim)
        logvar = self.fc_logvar(h)  # (batch, latent_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization
        return z, mu, logvar


class LSTDShortEncoder(nn.Module):
    """
        Short-term state encoder using fully-connected layers.
        Produces a short-term latent variable (mean and logvar) from input sequence.
    """

    def __init__(self, input_vars: int, seq_len: int, hidden_dim: int, latent_dim: int):
        super(LSTDShortEncoder, self).__init__()
        # Flatten input and use MLP to extract short-term features
        self.fc1 = nn.Linear(input_vars * seq_len, hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x: Tensor of shape (batch_size, seq_len, input_vars)
        :return: z_s (sampled latent), mu_s, logvar_s
        """
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)  # (batch, input_vars*seq_len)
        h = self.act(self.fc1(x_flat))  # (batch, hidden_dim)
        mu = self.fc_mu(h)  # (batch, latent_dim)
        logvar = self.fc_logvar(h)  # (batch, latent_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization
        return z, mu, logvar


class LSTDLatentTransition(nn.Module):
    """
    Latent transition module: predicts future latent states from current latents.
    """

    def __init__(self, latent_dim_d: int, latent_dim_s: int, pred_len: int, hidden_dim: int):
        super(LSTDLatentTransition, self).__init__()
        # Long-term latent prediction MLP
        self.fc_d1 = nn.Linear(latent_dim_d, hidden_dim)
        self.fc_d2 = nn.Linear(hidden_dim, latent_dim_d * pred_len)
        # Short-term latent prediction MLP
        self.fc_s1 = nn.Linear(latent_dim_s, hidden_dim)
        self.fc_s2 = nn.Linear(hidden_dim, latent_dim_s * pred_len)
        self.act = nn.LeakyReLU()
        self.pred_len = pred_len
        self.latent_dim_d = latent_dim_d
        self.latent_dim_s = latent_dim_s

    def forward(self, z_d: torch.Tensor, z_s: torch.Tensor):
        """
        :param z_d: long-term latent tensor (batch_size, latent_dim_d)
        :param z_s: short-term latent tensor (batch_size, latent_dim_s)
        :return: z_d_future (batch_size, pred_len, latent_dim_d),
                 z_s_future (batch_size, pred_len, latent_dim_s)
        """
        # Long-term latent transition
        h_d = self.act(self.fc_d1(z_d))
        out_d = self.fc_d2(h_d)  # (batch, latent_dim_d * pred_len)
        z_d_future = out_d.view(-1, self.pred_len, self.latent_dim_d)
        # Short-term latent transition
        h_s = self.act(self.fc_s1(z_s))
        out_s = self.fc_s2(h_s)  # (batch, latent_dim_s * pred_len)
        z_s_future = out_s.view(-1, self.pred_len, self.latent_dim_s)
        return z_d_future, z_s_future


class LSTDDecoder(nn.Module):
    """
    Decoder module: maps combined long-term and short-term latents to final forecasts.
    """

    def __init__(self, latent_dim_d: int, latent_dim_s: int, pred_len: int, output_vars: int, hidden_dim: int):
        super(LSTDDecoder, self).__init__()
        # Calculate input dimension: (z_d + z_s) for historical + (predicted z_d + z_s) for future
        input_dim = latent_dim_d * (1 + pred_len) + latent_dim_s * (1 + pred_len)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, pred_len * output_vars)
        self.pred_len = pred_len
        self.output_vars = output_vars

    def forward(self, z_d: torch.Tensor, z_s: torch.Tensor,
                z_d_future: torch.Tensor, z_s_future: torch.Tensor):
        """
        :param z_d: long-term latent (batch_size, latent_dim_d)
        :param z_s: short-term latent (batch_size, latent_dim_s)
        :param z_d_future: predicted long-term latents (batch_size, pred_len, latent_dim_d)
        :param z_s_future: predicted short-term latents (batch_size, pred_len, latent_dim_s)
        :return: out: predictions tensor of shape (batch_size, pred_len, output_vars)
        """
        batch_size = z_d.size(0)
        # Flatten future latents
        z_d_future_flat = z_d_future.reshape(batch_size, -1)
        z_s_future_flat = z_s_future.reshape(batch_size, -1)
        # Concatenate all latent information
        latent_concat = torch.cat([z_d, z_s, z_d_future_flat, z_s_future_flat], dim=1)
        h = self.act(self.fc1(latent_concat))
        out = self.fc2(h)  # (batch, pred_len * output_vars)
        out = out.view(batch_size, self.pred_len, self.output_vars)
        return out


class LSTDPriorNetwork(nn.Module):
    """
    Prior network for a latent variable: outputs a scalar (e.g., log-determinant) from latent.
    """

    def __init__(self, latent_dim: int):
        super(LSTDPriorNetwork, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)
        self.act = nn.LeakyReLU()

    def forward(self, z: torch.Tensor):
        """
        :param z: latent tensor (batch_size, latent_dim)
        :return: scalar tensor (batch_size, 1), e.g., log-det of Jacobian
        """
        h = self.act(self.fc1(z))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        log_det = self.fc4(h)
        return log_det


class LSTD(nn.Module):
    """
        Ruichu Cai, Haiqin Huang, Zhifang Jiang, Zijian Li, Changze Zhou, Yuequn Liu, Yuming Liu, Zhifeng Hao.
        LSTD: Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting,
        AAAI 2025.

        TODO: code review according to the paper. 2025-05-02

        This version is wrong.

        Original code: https://github.com/DMIRLAB-Group/LSTD/

        :param input_vars: int, number of input features.
        :param output_window_size: int, forecasting horizon (prediction length).
        :param output_vars: int, number of output variables.
        :param input_window_size: int, length of input sequence.
        :param label_window_size: int, length of label/input to decoder (if used).
        :param latent_dim_d: int, dimension of long-term latent state.
        :param latent_dim_s: int, dimension of short-term latent state.
        :param hidden_dim: int, hidden dimension for neural layers.
    """

    def __init__(self,
                 input_window_size: int = 60,
                 input_vars: int = 1,
                 output_window_size: int = 1,
                 output_vars: int = 1,
                 label_window_size: int = 0,
                 latent_dim_d: int = 16,
                 latent_dim_s: int = 16,
                 hidden_dim: int = 512,
                 dropout_rate: float = 0.1):
        super(LSTD, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size
        self.output_vars = output_vars
        self.label_window_size = label_window_size
        self.latent_dim_d = latent_dim_d
        self.latent_dim_s = latent_dim_s

        # Long-term and short-term encoders
        self.long_encoder = LSTDLongEncoder(input_vars=self.input_vars,
                                            hidden_dim=hidden_dim,
                                            latent_dim=latent_dim_d)
        self.short_encoder = LSTDShortEncoder(input_vars=self.input_vars,
                                              seq_len=self.input_window_size,
                                              hidden_dim=hidden_dim,
                                              latent_dim=latent_dim_s)
        # Latent transition module
        self.latent_transition = LSTDLatentTransition(latent_dim_d=latent_dim_d,
                                                      latent_dim_s=latent_dim_s,
                                                      pred_len=self.output_window_size,
                                                      hidden_dim=hidden_dim)
        # Prior networks for long-term and short-term latents
        self.prior_long = LSTDPriorNetwork(latent_dim=latent_dim_d)
        self.prior_short = LSTDPriorNetwork(latent_dim=latent_dim_s)
        # Decoder to map latents to predictions
        self.decoder = LSTDDecoder(latent_dim_d=latent_dim_d,
                                   latent_dim_s=latent_dim_s,
                                   pred_len=self.output_window_size,
                                   output_vars=self.output_vars,
                                   hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor):
        """
            :param x: Tensor of shape ``(batch_size, seq_len, input_vars)``
            :return: out: Tensor of shape ``(batch_size, pred_len, output_vars)``
        """

        # Encode input to latent distributions
        z_d, mu_d, logvar_d = self.long_encoder(x)  # long-term latent
        z_s, mu_s, logvar_s = self.short_encoder(x)  # short-term latent

        # Latent state transition for future latents
        z_d_future, z_s_future = self.latent_transition(z_d, z_s)
        # Compute prior network outputs (e.g., for KL losses)
        prior_d = self.prior_long(z_d)
        prior_s = self.prior_short(z_s)

        # Decode latents to forecast output
        out = self.decoder(z_d, z_s, z_d_future, z_s_future)

        return out
