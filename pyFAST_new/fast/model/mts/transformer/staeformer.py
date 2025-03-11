#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """
        Perform attention across the -2 dim (the -1 dim is `model_dim`).
        Make sure the tensor is permuted to correct shape before attention.
        E.g.
        - Input shape (batch_size, in_steps, num_nodes, model_dim).
        - Then the attention will be performed across the nodes.
        Also, it supports different src and tgt length.
        But must `src length == K length == V length`.
    """

    def __init__(self, model_dim: int, num_heads: int = 8, mask: bool = False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (query @ key) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(tgt_length, src_length, dtype=torch.bool, device=query.device).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    """
        Self-attention layer.
    """

    def __init__(self, model_dim: int, feed_forward_dim: int = 2048, num_heads: int = 8, dropout: float = 0, mask: bool = False):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int = -2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    """
        Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng, Jinliang Deng, Quanjun Chen, Xuan Song
        STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting
        url: https://arxiv.org/abs/2308.10425
        Official Code: https://github.com/XDZhelheim/STAEformer

        :param input_window_size: input window size.
        :param input_vars: number of input variables.
        :param output_window_size: output window size.
        :param input_dim: input feature dimension.
        :param output_dim: output feature dimension.
        :param input_embedding_dim: input embedding dimension.
        :param tod_steps_per_day: number of time steps per day (e.g., 24 for hourly data).
        :param tod_embedding_dim: time-of-day embedding dimension.
        :param dow_embedding_dim: day-of-week embedding dimension.
        :param spatial_embedding_dim: patial embedding dimension
        :param adaptive_embedding_dim: adaptive embedding dimension
        :param dim_ff: feed forward dimension.
        :param num_heads: number of attention heads.
        :param num_layers: number of encoder layers.
        :param dropout_rate: dropout rate.
        :param use_mixed_proj: whether to use mixed projection.
    """

    def __init__(self, input_window_size: int = 1, input_vars: int = 1, output_window_size: int = 1,
                 input_dim: int = 1, output_dim: int = 1, input_embedding_dim: int = 24,
                 tod_steps_per_day: int = 24, tod_embedding_dim: int = 0, dow_embedding_dim: int = 0,
                 spatial_embedding_dim: int = 0, adaptive_embedding_dim: int = 80,
                 dim_ff: int = 256, num_heads: int = 4, num_layers: int = 3,
                 dropout_rate: float = 0.1, use_mixed_proj: bool = True):
        super(STAEformer, self).__init__()
        self.input_window_size = input_window_size
        self.input_vars = input_vars
        self.output_window_size = output_window_size

        self.in_steps = self.input_window_size
        self.num_nodes = self.input_vars
        self.out_steps = self.output_window_size

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.tod_steps_per_day = tod_steps_per_day
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + spatial_embedding_dim
                + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(tod_steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(self.in_steps, self.num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                self.in_steps * self.model_dim, self.out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(self.in_steps, self.out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, dim_ff, num_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, dim_ff, num_heads, dropout_rate)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        """ x -> [batch_size, input_window_size(in_steps), input_vars(num_nodes), 3(input_dim+tod+dow)] """

        x = x.unsqueeze(-1)  # -> [batch_size, input_window_size, input_vars, 1]

        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]  # -> [batch_size, input_window_size(in_steps), input_vars(num_nodes), 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]  # -> [batch_size, input_window_size(in_steps), input_vars(num_nodes), 1]

        x = x[..., : self.input_dim]  # -> [batch_size, input_window_size(in_steps), input_vars(num_nodes), input_dim]

        x = self.input_proj(x)  # -> [batch_size, input_window_size(in_steps), input_vars(num_nodes), input_embedding_dim]
        features = [x]  # -> list([batch_size, input_window_size(in_steps), input_vars(num_nodes), input_embedding_dim])
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding( (tod * self.tod_steps_per_day).long())  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(dow.long())  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(batch_size, self.in_steps, *self.node_emb.shape)
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(size=(batch_size, *self.adaptive_embedding.shape))
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # -> [batch_size, input_window_size(in_steps), input_vars(num_nodes), model_dim]

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # -> [batch_size, input_vars(num_nodes), input_window_size(in_steps), model_dim]
            out = out.reshape(batch_size, self.num_nodes, self.in_steps * self.model_dim)  # -> [batch_size, input_vars(num_nodes), input_window_size(in_steps) * model_dim]
            out = self.output_proj(out).view(batch_size, self.num_nodes, self.out_steps, self.output_dim)  # -> [batch_size, input_vars(num_nodes), output_window_size(out_steps), output_dim]
            out = out.transpose(1, 2)  # -> [batch_size, output_window_size(out_steps), input_vars(num_nodes), output_dim]
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(out)  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(out.transpose(1, 3))  # (batch_size, out_steps, num_nodes, output_dim)

        out = out.squeeze(-1)  # ->  [batch_size, output_window_size(out_steps), input_vars(num_nodes)]
        return out
