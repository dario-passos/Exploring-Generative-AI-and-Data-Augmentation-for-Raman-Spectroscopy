"""
Keras implementation of the VAE architectures used in notebooks.

Encoder: Dense [2000, 1000, 500] -> mean, logvar heads
Decoder: Dense [500, 1000, 2000] -> output_dim

Includes a `VAE` model with custom train_step computing reconstruction + KL losses.
"""
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_encoder(input_dim: int, latent_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="x")
    x = layers.Dense(2000, activation="relu")(inputs)
    x = layers.Dense(1000, activation="relu")(x)
    x = layers.Dense(500, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_logvar = layers.Dense(latent_dim, name="z_logvar")(x)
    return keras.Model(inputs, [z_mean, z_logvar], name="encoder")


def build_decoder(latent_dim: int, output_dim: int) -> keras.Model:
    inputs = keras.Input(shape=(latent_dim,), name="z")
    x = layers.Dense(500, activation="relu")(inputs)
    x = layers.Dense(1000, activation="relu")(x)
    x = layers.Dense(2000, activation="relu")(x)
    outputs = layers.Dense(output_dim, name="x_recon")(x)  # linear; pair with MSE/MAE
    return keras.Model(inputs, outputs, name="decoder")


class Sampling(layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        z_mean, z_logvar = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps


class VAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, recon_loss: str = "mse") -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()
        if recon_loss == "mae":
            self._recon = keras.losses.MeanAbsoluteError()
        else:
            self._recon = keras.losses.MeanSquaredError()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):  # type: ignore[override]
        return [self.total_loss_tracker, self.recon_loss_tracker, self.kl_loss_tracker]

    def call(self, x, training=False):  # type: ignore[override]
        z_mean, z_logvar = self.encoder(x, training=training)
        z = self.sampling((z_mean, z_logvar))
        x_hat = self.decoder(z, training=training)
        return x_hat

    def train_step(self, x):  # type: ignore[override]
        if isinstance(x, (tuple, list)):
            x = x[0]
        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(x, training=True)
            z = self.sampling((z_mean, z_logvar))
            x_hat = self.decoder(z, training=True)
            recon_loss = self._recon(x, x_hat)
            # KL divergence: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
            kl = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
            total = recon_loss + kl
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.total_loss_tracker.update_state(total)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl)
        return {"loss": self.total_loss_tracker.result(), "recon": self.recon_loss_tracker.result(), "kl": self.kl_loss_tracker.result()}


def build_vae(input_dim: int, latent_dim: int, recon_loss: str = "mse") -> VAE:
    enc = build_encoder(input_dim, latent_dim)
    dec = build_decoder(latent_dim, input_dim)
    return VAE(enc, dec, recon_loss=recon_loss)

