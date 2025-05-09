#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
        d_model : La dimension de l'espace de représentation des embeddings d'entrée
        et de sortie.

        num_heads : Le nombre de têtes d'attention. Chaque tête d'attention apprend
        à capturer différents aspects des dépendances dans la séquence d'entrée.

        depth : La dimension de chaque tête d'attention, calculée comme d_model // num_heads.
        Ceci assure que la concaténation des sorties de toutes les têtes donne une
        représentation de dimension d_model.

        Les transformations linéaires wq, wk, et wv sont utilisées pour générer les
        matrices de requêtes (queries), de clés (keys), et de valeurs (values)
        respectivement. La transformation dense est appliquée après l'opération
        d'attention pour obtenir la sortie finale.
    """
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Cette fonction réorganise les dimensions du tenseur x afin de séparer les têtes d'attention :

        x.view(batch_size, -1, self.num_heads, self.depth) : Réorganise le tenseur
        pour avoir num_heads têtes avec depth dimensions chacune. La nouvelle forme
        est [batch_size, seq_length, num_heads, depth].

        permute(0, 2, 1, 3) : Permute les dimensions pour obtenir la forme
        [batch_size, num_heads, seq_length, depth], ce qui est nécessaire pour
        le calcul de l'attention multi-tête.
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, mask=None):
        """
        Les transformations linéaires wq, wk, et wv sont appliquées à l'entrée x
        pour obtenir les requêtes, clés, et valeurs, qui sont ensuite réorganisées
        pour les têtes d'attention.
        """

        batch_size = x.size(0)

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        """
        Les scores d'attention sont calculés en prenant le produit scalaire des
        requêtes et des clés, normalisé par la racine carrée de la profondeur.
        Cette normalisation aide à stabiliser les gradients lors de l'entraînement.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            scores += (mask * -1e9)

        attn_weights = scores.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        output = self.dense(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x, mask=None):
        x = self.dropout(torch.relu(self.linear1(x)))
        x = self.linear2(x)

        return x


class LayerNormResidual(nn.Module):
    def __init__(self, d_model, sublayer):
        super(LayerNormResidual, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)  # Linear transformation for residual connection

    def forward(self, x, mask=None):
        normalized_x = self.norm(x)
        sublayer_output = self.sublayer(normalized_x, mask)

        # Apply linear transformation for residual connection
        residual = self.linear(normalized_x)

        # Ensure residual and sublayer_output have the same shape
        if residual.size(1) != sublayer_output.size(1):
            raise RuntimeError("Size mismatch between residual and sublayer_output.")

        # Add residual connection
        output = residual + sublayer_output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.attention = LayerNormResidual(d_model, MultiHeadAttention(d_model, num_heads))
        self.feed_forward = LayerNormResidual(d_model, FeedForward(d_model, d_ff, dropout))

    def forward(self, x, mask=None):
        x = self.attention(x, mask)
        x = self.feed_forward(x, mask)
        return x


class TimesFM(nn.Module):
    """
        Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou.
        A Decoder-Only Foundation Model for Time-Series Forecasting, ICML 2024.
        url: https://arxiv.org/pdf/2310.10688

        :param input_vars: input variables number.
        :param output_window_size: output window size.
        :param d_model: model dimension, a.k.a., embedding size.
        :param num_heads: head number, a.k.a., attention head number.
        :param num_layers: number of decoder layers.
        :param dim_ff: feed forward dimension.
        :param dropout_rate: dropout rate.
    """

    def __init__(self, input_vars: int, output_window_size: int,
                 d_model: int, num_heads: int, num_layers: int, dim_ff: int, dropout_rate: float = 0.1):
        super(TimesFM, self).__init__()
        self.input_size = input_vars
        self.output_window_size = output_window_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_ff = dim_ff
        self.dropout_rate = dropout_rate

        self.embedding = nn.Linear(input_vars, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 5000, d_model))
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, num_heads, dim_ff, dropout_rate) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, input_vars)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.embedding(x)
        x += self.pos_embedding[:, :x.shape[1], :]

        for idx, layer in enumerate(self.layers):
            x = layer(x, mask)

        out = self.output_layer(x)  # -> (batch_size, input_window_size, input_size)
        out = out[:, -self.output_window_size:, :]  # -> (batch_size, output_window_size, input_size)
        return out


def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask
