#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from fast.data import InstanceScale, InstanceStandardScale


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        """
        :param z_mean: shape is ``(batch, seq, latent_dim)``
        :param z_log_var: shape is ``(batch, seq, latent_dim)``
        """
        epsilon = torch.randn(z_mean.shape).to(z_mean.device)

        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class TemporalVAE(nn.Module):
    """
        Temporal Conditional Variational Autoencoder.
    """

    def __init__(self, feature_dim, condition_dim, embedding_dim, latent_dim, num_heads, dropout_rate=0.,
                 use_instance_scale: bool = True):
        super(TemporalVAE, self).__init__()

        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.feature_embedding_layer = nn.Linear(feature_dim, embedding_dim)
        self.condition_embedding_layer = nn.Linear(condition_dim, embedding_dim)

        # Encoder
        self.encode_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout_rate, batch_first=True)

        self.mean_layer = nn.Linear(embedding_dim, latent_dim)
        self.log_var_layer = nn.Linear(embedding_dim, latent_dim)

        self.sample_layer = Sampling()

        # Decoder
        self.sample_embedding_layer = nn.Linear(latent_dim, embedding_dim)
        self.decode_attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout_rate, batch_first=True)
        self.l1 = nn.Linear(embedding_dim, feature_dim)

        self.inst_scale = InstanceStandardScale(feature_dim) if use_instance_scale else InstanceScale()

    def encode(self, features, condition):
        """
            :param features: shape is ``(batch, seq_len, feature_dim)``
            :param condition: shape is ``(batch, seq_len, condition_dim)``
        """
        # Feature Encoding
        features_embedding = self.feature_embedding_layer(features)  # -> (batch_size, seq_len, embedding_dim)
        features_embedding = features_embedding.relu()

        # Condition Embedding
        condition_embedding = self.condition_embedding_layer(condition)  # -> (batch_size, seq_len, embedding_dim)
        # condition_embedding = condition_embedding.relu()

        # e_out -> (batch_size, seq_len, embedding_dim)
        e_out, _ = self.encode_attention(features_embedding, condition_embedding, condition_embedding)

        return e_out

    def decode(self, samples, condition):
        """
            :param samples: shape is ``(batch, seq_len, latent_dim)``
            :param condition: shape is ``(batch, seq_len, condition_dim)``
        """
        # Samples Embedding
        sample_embedding = self.sample_embedding_layer(samples)  # -> (batch_size, seq_len, embedding_dim)
        sample_embedding = sample_embedding.relu()

        # Condition Embedding
        condition_embedding = self.condition_embedding_layer(condition)  # -> (batch_size, seq_len, embedding_dim)
        # condition_embedding = condition_embedding.relu()

        # Attention -> (batch_size, seq_len, embedding_dim)
        d_out, _ = self.decode_attention(sample_embedding, condition_embedding, condition_embedding)

        # Reconstruction
        d_out = self.l1(d_out)  # -> (batch_size, seq_len, feature_dim)
        # d_out = d_out.sigmoid()

        return d_out

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        :param x: shape is ``(batch, seq_len, feature_dim)``
        :param condition: shape is ``(batch, seq_len, condition_dim)``
        """
        # x, condition = condition, x

        x = self.inst_scale.fit_transform(x)

        encoder_out = self.encode(x, condition)  # -> (batch_size, seq_len, embedding_dim)

        z_mean = self.mean_layer(encoder_out)  # -> (batch_size, seq_len, latent_dim)
        z_log_var = self.log_var_layer(encoder_out)  # -> (batch_size, seq_len, latent_dim)

        # Sampling
        samples = self.sample_layer(z_mean, z_log_var)  # -> (batch_size, seq_len, latent_dim)

        generated_x = self.decode(samples, condition)  # -> (batch_size, seq_len, feature_dim)

        generated_x = self.inst_scale.inverse_transform(generated_x)

        return generated_x

    def generate(self, condition: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape is ``(batch, seq_len, feature_dim)``, not be used here.
        :param condition: shape is ``(batch, seq_len, condition_dim)``, its device should be the same as the model.
        """

        # Sampling -> (batch_size, seq_len, latent_dim)
        samples = torch.randn((*condition.shape[:-1], self.latent_dim), dtype=condition.dtype).to(condition.device)
        generated_x = self.decode(samples, condition)  # -> (batch_size, seq_len, feature_dim)

        return generated_x
