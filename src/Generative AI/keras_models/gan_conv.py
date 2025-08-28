"""
Keras implementation of a 1D-convolutional GAN similar to the PyTorch notebook.

Generator uses Dense -> reshape -> series of Conv1DTranspose (or UpSampling1D+Conv1D)
to synthesize a 1-channel spectral vector. Finally Flatten + Dense ensures exact output length.

Discriminator mirrors the Conv1D stack with strides, ending in Dense -> 1 logit.

Note: Conv1DTranspose is available in TF 2.12+. If not available, a fallback using
UpSampling1D + Conv1D can be swapped in by replacing Conv1DTranspose blocks.
"""
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _conv1d_transpose(x, filters: int, kernel_size: int, strides: int, padding: str = "same", use_bias: bool = False):
    # Use Conv1DTranspose if available, else UpSampling1D + Conv1D fallback
    if hasattr(layers, "Conv1DTranspose"):
        return layers.Conv1DTranspose(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    else:
        x = layers.UpSampling1D(size=strides)(x)
        return layers.Conv1D(filters, kernel_size, padding=padding, use_bias=use_bias)(x)


def build_generator_conv1d(output_len: int, noise_dim: int = 128, base_width: int = 64) -> keras.Model:
    """
    Build a 1D ConvTranspose-based generator. The exact intermediate sequence
    mirrors the PyTorch example in spirit; final Dense ensures length == output_len.
    """
    inputs = keras.Input(shape=(noise_dim,), name="z")
    # Project and reshape to (length, channels)
    init_len = max(output_len // 64, 16)  # heuristic to start small and upsample
    proj_units = init_len * base_width
    x = layers.Dense(proj_units, use_bias=False)(inputs)
    x = layers.ReLU()(x)
    x = layers.Reshape((init_len, base_width))(x)

    # Series of upsampling transpose convs
    x = _conv1d_transpose(x, 256, kernel_size=4, strides=2)
    x = layers.ReLU()(x)
    x = _conv1d_transpose(x, 128, kernel_size=4, strides=2)
    x = layers.ReLU()(x)
    x = _conv1d_transpose(x, 64, kernel_size=4, strides=2)
    x = layers.ReLU()(x)
    x = _conv1d_transpose(x, 32, kernel_size=4, strides=2)
    x = layers.ReLU()(x)
    x = _conv1d_transpose(x, 16, kernel_size=4, strides=2)
    x = layers.ReLU()(x)
    x = _conv1d_transpose(x, 8, kernel_size=3, strides=1, padding="same")
    x = layers.ReLU()(x)
    x = layers.Conv1D(1, kernel_size=3, padding="same", use_bias=False)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(output_len, name="x_gen")(x)  # linear output
    return keras.Model(inputs, outputs, name="generator_conv1d")


def build_discriminator_conv1d(input_len: int, leaky_alpha: float = 0.2) -> keras.Model:
    inputs = keras.Input(shape=(input_len,), name="x")
    x = layers.Reshape((input_len, 1))(inputs)

    x = layers.Conv1D(8, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Conv1D(16, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Conv1D(32, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Conv1D(64, kernel_size=3, strides=2, padding="same", use_bias=False)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    logits = layers.Dense(1, name="logits")(x)  # linear logit for BCE(from_logits=True)
    return keras.Model(inputs, logits, name="discriminator_conv1d")

