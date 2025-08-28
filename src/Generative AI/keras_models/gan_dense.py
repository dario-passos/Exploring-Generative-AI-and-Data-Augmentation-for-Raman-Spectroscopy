"""
Keras implementation of the fully-connected GAN used in the notebooks.

This mirrors the PyTorch architecture:
- Generator: Dense layers with ReLU, final linear to output length
- Discriminator: Dense layers with LeakyReLU, final linear logit (use BCE(from_logits=True))

Usage example:
    gen = build_generator_dense(output_dim=3276, noise_dim=128)
    disc = build_discriminator_dense(input_dim=3276)

Note: Keep data scaling consistent with your loss choice.
      With tf.keras.losses.BinaryCrossentropy(from_logits=True),
      the discriminator output should be linear (no sigmoid), which this matches.
"""
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_generator_dense(output_dim: int, noise_dim: int = 128) -> keras.Model:
    inputs = keras.Input(shape=(noise_dim,), name="z")
    x = layers.Dense(256)(inputs)
    x = layers.ReLU()(x)
    x = layers.Dense(512)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(1024)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(2048)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(output_dim, name="x_gen")(x)  # linear output (match logits-based training)
    return keras.Model(inputs, outputs, name="generator_dense")


def build_discriminator_dense(input_dim: int, leaky_alpha: float = 0.2) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="x")
    x = layers.Dense(2048)(inputs)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    x = layers.Dense(64)(x)
    x = layers.LeakyReLU(alpha=leaky_alpha)(x)
    logits = layers.Dense(1, name="logits")(x)  # linear logit
    return keras.Model(inputs, logits, name="discriminator_dense")


class DenseGAN(keras.Model):
    """Optional: end-to-end GAN wrapper with custom train_step.

    This class is provided for convenience if you prefer a single compiled model.
    It follows the common Keras GAN pattern.
    """

    def __init__(
        self,
        generator: keras.Model,
        discriminator: keras.Model,
        noise_dim: int = 128,
        d_steps: int = 1,
    ) -> None:
        super().__init__()
        self.gen = generator
        self.disc = discriminator
        self.noise_dim = noise_dim
        self.d_steps = d_steps

    def compile(self, g_optimizer, d_optimizer, bce_loss=None, **kwargs):  # type: ignore[override]
        super().compile(**kwargs)
        self.g_opt = g_optimizer
        self.d_opt = d_optimizer
        self.bce = bce_loss or keras.losses.BinaryCrossentropy(from_logits=True)
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")

    @property
    def metrics(self):  # type: ignore[override]
        return [self.g_loss_tracker, self.d_loss_tracker]

    def train_step(self, real_x):  # type: ignore[override]
        if isinstance(real_x, (tuple, list)):
            real_x = real_x[0]

        batch_size = tf.shape(real_x)[0]

        # Update discriminator
        for _ in range(self.d_steps):
            z = tf.random.normal(shape=(batch_size, self.noise_dim))
            with tf.GradientTape() as tape:
                fake_x = self.gen(z, training=True)
                real_logits = self.disc(real_x, training=True)
                fake_logits = self.disc(fake_x, training=True)
                d_loss_real = self.bce(tf.ones_like(real_logits), real_logits)
                d_loss_fake = self.bce(tf.zeros_like(fake_logits), fake_logits)
                d_loss = d_loss_real + d_loss_fake
            grads = tape.gradient(d_loss, self.disc.trainable_variables)
            self.d_opt.apply_gradients(zip(grads, self.disc.trainable_variables))

        # Update generator
        z = tf.random.normal(shape=(batch_size, self.noise_dim))
        with tf.GradientTape() as tape:
            fake_x = self.gen(z, training=True)
            fake_logits = self.disc(fake_x, training=False)
            g_loss = self.bce(tf.ones_like(fake_logits), fake_logits)
        grads = tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.gen.trainable_variables))

        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

