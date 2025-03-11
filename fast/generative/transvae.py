#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn

from fast.data import InstanceScale, InstanceStandardScale
from fast.model.base import get_activation_cls


class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        """
        :param z_mean: shape is ``(batch, seq, latent_dim)``
        :param z_log_var: shape is ``(batch, seq, latent_dim)``
        """
        epsilon = torch.randn(z_mean.shape).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


class TransVAE(nn.Module):
    """
        Transformer-based Conditional Variational Autoencoder.

        :param feature_dim: feature dimension.
        :param condition_dim: condition dimension.
        :param latent_dim: the dimension of latent variable.
        :param d_model: the dimension of model.
        :param num_heads: the number of heads in multi-head attention.
        :param num_layers: the number of layers in encoder and decoder.
        :param dropout_rate: the dropout rate.
        :param activation: the activation function in encoder/decoder.
        :param use_instance_scale: whether to use instance standard scale (a.k.a., RevIN).
    """

    def __init__(self, feature_dim: int, condition_dim: int, latent_dim: int,
                 d_model: int, num_heads: int, num_layers: int = 1, dropout_rate: float = 0.,
                 activation: str = 'relu', use_instance_scale: bool = True):
        super(TransVAE, self).__init__()

        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim  # z_dim, the dimension of latent variable
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_ff = 4 * d_model
        self.num_layers = num_layers

        self.activation = activation

        self.feature_embedding_layer = nn.Linear(self.feature_dim, self.d_model)
        self.condition_embedding_layer = nn.Linear(self.condition_dim, self.d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, self.num_heads, self.dim_ff,
                                                   dropout_rate, self.activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.mean_layer = nn.Linear(self.d_model, latent_dim)
        self.log_var_layer = nn.Linear(self.d_model, latent_dim)

        self.sample_layer = Sampling()

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.num_heads, self.dim_ff,
                                                   dropout_rate, self.activation, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)

        self.sample_embedding_layer = nn.Linear(self.latent_dim, self.d_model)
        self.output_layer = nn.Linear(self.d_model, self.feature_dim)

        self.inst_scale = InstanceStandardScale(self.feature_dim) if use_instance_scale else InstanceScale()

    def encode(self, features, condition):
        """
            :param features: shape is ``(batch, seq_len, feature_dim)``
            :param condition: shape is ``(batch, seq_len, condition_dim)``
        """
        # Feature Encoding
        features_embedding = self.feature_embedding_layer(features) # .relu()
        condition_embedding = self.condition_embedding_layer(condition) # .relu()

        # Combine features and condition embeddings
        encoder_input = features_embedding + condition_embedding

        # Transformer Encoder
        encoded_output = self.encoder(encoder_input)  # -> (batch_size, seq_len, embedding_dim)

        return encoded_output

    def decode(self, samples, condition):
        """
            :param samples: shape is ``(batch, seq_len, latent_dim)``
            :param condition: shape is ``(batch, seq_len, condition_dim)``
        """
        # Samples Embedding
        sample_embedding = self.sample_embedding_layer(samples) # .relu()
        condition_embedding = self.condition_embedding_layer(condition) # .relu()

        # Transformer Decoder
        decoder_output = self.decoder(tgt=sample_embedding, memory=condition_embedding)

        # Reconstruction
        reconstructed_output = self.output_layer(decoder_output)  # -> (batch_size, seq_len, feature_dim)

        return reconstructed_output

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        https://github.com/oriondollar/TransVAE/blob/master/transvae/trans_models.py
        :param x: shape is ``(batch, seq_len, feature_dim)``
        :param condition: shape is ``(batch, seq_len, condition_dim)``
        """
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
        :param condition: shape is ``(batch, seq_len, condition_dim)``, its device should be the same as the model.
        """
        # Sampling -> (batch_size, seq_len, latent_dim)
        samples = torch.randn((*condition.shape[:-1], self.latent_dim), dtype=condition.dtype).to(condition.device)

        generated_x = self.decode(samples, condition)  # -> (batch_size, seq_len, feature_dim)

        return generated_x
