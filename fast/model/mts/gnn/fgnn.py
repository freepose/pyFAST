#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierGNN(nn.Module):
    """
        Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu
        FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective,
        NIPS 2023

        url: https://arxiv.org/pdf/2311.06190
        Official Code: https://github.com/aikunyi/FourierGNN/tree/main

        :param input_window_size: number of input variables.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param embed_size: embedding dimension size.
        :param hidden_size: hidden layer dimension size.
        :param hard_thresholding_fraction: sparsity control parameter.
        :param hidden_size_factor: hidden layer expansion factor.
        :param sparsity_threshold: soft thresholding intensity.
    """

    def __init__(self, input_window_size: int, input_vars: int, output_window_size: int,
                 embed_size: int = 128, hidden_size: int = 256, hard_thresholding_fraction: float = 1,
                 hidden_size_factor: int = 1, sparsity_threshold: float = 0.01):
        super(FourierGNN, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold

        self.number_frequency = 1
        self.frequency_size = self.embed_size // self.number_frequency
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size,
                                     self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(
            self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.input_window_size, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_window_size)
        )

    def tokenEmb(self, x: torch.Tensor):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x: torch.Tensor, B: int, N: int, L: int):
        o1_real = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o1_imag = torch.zeros([B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
                              device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum('bli,ii->bli', x.real, self.w1[0]) - \
            torch.einsum('bli,ii->bli', x.imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag = F.relu(
            torch.einsum('bli,ii->bli', x.imag, self.w1[0]) + \
            torch.einsum('bli,ii->bli', x.real, self.w1[1]) + \
            self.b1[1]
        )

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum('bli,ii->bli', o1_real, self.w2[0]) - \
            torch.einsum('bli,ii->bli', o1_imag, self.w2[1]) + \
            self.b2[0]
        )

        o2_imag = F.relu(
            torch.einsum('bli,ii->bli', o1_imag, self.w2[0]) + \
            torch.einsum('bli,ii->bli', o1_real, self.w2[1]) + \
            self.b2[1]
        )

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
            torch.einsum('bli,ii->bli', o2_real, self.w3[0]) - \
            torch.einsum('bli,ii->bli', o2_imag, self.w3[1]) + \
            self.b3[0]
        )

        o3_imag = F.relu(
            torch.einsum('bli,ii->bli', o2_imag, self.w3[0]) + \
            torch.einsum('bli,ii->bli', o2_real, self.w3[1]) + \
            self.b3[1]
        )

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)  # x -> [bs, var*seq]
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)  # x -> [bs, var*seq, embed_size]

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')  # x -> [bs, var*seq, embed_size]

        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)  # x -> [bs, var*seq//2+1, frequency_size]

        bias = x  # bias -> [bs, var*seq//2+1, frequency_size]

        # FourierGNN
        x = self.fourierGC(x, B, N, L)  # x -> [bs, var*seq//2+1, frequency_size]

        x = x + bias

        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)  # x -> [bs, var*seq//2+1, embed_size]

        # ifft
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")  # x -> [bs, var*seq, embed_size]

        x = x.reshape(B, N, L, self.embed_size)  # x -> [bs, var, seq, embed_size]
        x = x.permute(0, 1, 3, 2)  # x -> [bs, var, embed_size, seq]

        # projection
        x = torch.matmul(x, self.embeddings_10)  # x -> [bs, var, embed_size, 8]
        x = x.reshape(B, N, -1)  # x -> [bs, var, embed_size*8]
        x = self.fc(x)  # x -> [bs, var, pre_length]

        x = x.permute(0, 2, 1)  # x -> [bs, pre_length, var]
        return x
