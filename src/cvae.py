import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pickle
import os

# ==================== Sampling Functions ==================== #
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = k.backend.shape(z_mean)[0]
    dim = k.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = k.backend.random_normal(shape=(batch, dim))
    return z_mean + k.backend.exp(0.5 * z_log_var) * epsilon

# ==================== VAE Model ==================== #
class VAE(tf.keras.Model):
    def __init__(self, encoder_net, decoder_net, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder_model = encoder_net
        self.decoder_model = decoder_net
        
    def compile(self, enc_optimizer, dec_optimizer):
        super(VAE, self).compile(metrics=["mse"])
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer

    def call(self, x):
        return self.autoencode(x)
    
    def plot_models(self):
        tf.keras.utils.plot_model(self.encoder_model, to_file=os.path.join('model','{}.png'.format(self.encoder_model.name)), show_shapes=True, dpi=64)
        tf.keras.utils.plot_model(self.decoder_model, to_file=os.path.join('model','{}.png'.format(self.decoder_model.name)), show_shapes=True, dpi=64)
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def encode(self, x):
        return self.encoder_model(x)
    
    def decode(self, z):
        return self.decoder_model(z)
    
    def autoencode(self, x):
        return self.decode(self.encode(x)[2])
    
    def train_step(self, x):
        if isinstance(x, tuple):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
            
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
#             x_in_noisy = tf.keras.layers.GaussianNoise(stddev=0.005)(x_in)
            z_out = self.encode(x_in)
            xhat_sample = self.decode(z_out[2])
            
            reconstruction_loss = tf.reduce_mean(
                k.losses.mse(x_out, xhat_sample)
            )
            reconstruction_loss *= np.prod(x_out.shape[1:])
            
            z_mean = z_out[0]
            z_log_var = z_out[1]
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            
            total_loss = reconstruction_loss + kl_loss
        
        grads1 = tape1.gradient(total_loss, self.encoder_model.trainable_weights)
        self.enc_optimizer.apply_gradients(zip(grads1, self.encoder_model.trainable_weights))
        
        grads2 = tape2.gradient(total_loss, self.decoder_model.trainable_weights)
        self.dec_optimizer.apply_gradients(zip(grads2, self.decoder_model.trainable_weights))
                
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "mse_loss": tf.keras.losses.MeanSquaredError()(x_out, xhat_sample)
        }

# ==================== Create & Compile VAE model ==================== #
def get_vae_model(encoder_net, decoder_net, input_dim, latent_dim, plot_models=False, lr=0.0003):
    vae = VAE(encoder_net, decoder_net, input_dim, latent_dim)
    
    decoder_optimizer = k.optimizers.Adam(
        learning_rate=lr
    )
    encoder_optimizer = k.optimizers.Adam(
        learning_rate=lr
    )
    
    vae.compile(
        enc_optimizer=encoder_optimizer,
        dec_optimizer=decoder_optimizer,
    )
    
    if plot_models:
        vae.plot_models()
    return vae


def get_encoder_model(latent_dim, input_shape, n_encode_filters=256, first_kernel_size=3, k_sz=3):
    n_window, n_ch = input_shape
    
    in0 = layers.Input(shape=input_shape)
    x = in0
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(n_encode_filters,
                    kernel_size=3,
                    padding='same', activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(n_encode_filters,
                    kernel_size=3,
                    padding='same', activation='relu',
                    strides=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(n_encode_filters,
                    kernel_size=3,
                    padding='same', activation='relu',
                    strides=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(n_encode_filters,
                    kernel_size=3,
                    padding='same', activation='relu',
                    strides=2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(n_encode_filters,
                    kernel_size=3,
                    padding='same', activation='relu',
                    strides=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    hidden = layers.Dense(latent_dim, activation='relu')(x)
    hidden = layers.Dropout(0.5)(hidden)
    hidden = layers.Dense(latent_dim, activation='relu')(hidden)
    hidden = layers.BatchNormalization()(hidden)
    hidden = layers.Dropout(0.5)(hidden)
    
    z_mean = layers.Dense(latent_dim)(hidden)
    z_log_var = layers.Dense(latent_dim)(hidden)
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z_sample')([z_mean, z_log_var])
    z_out = [z_mean, z_log_var, z]

    encoder_net = Model(in0, z_out, name='encoder')
    return encoder_net


def get_decoder_model(latent_dim, output_shape, n_decode_filters=256, k_sz=3):
    
    n_window, n_ch = output_shape

    z_in = layers.Input(shape=latent_dim)
    x = z_in
    x = layers.Dense(latent_dim, activation=None)(x)
    x = layers.Dropout(0.5)(x)
    
#     y = layers.Dense(latent_dim, activation='relu')(x)
#     y = layers.Dense(latent_dim, activation='relu')(x)
#     x = layers.Add()([x, y])
#     x = layers.Dropout(0.5)(x)
    
#     y = layers.Dense(latent_dim, activation='relu')(x)
#     y = layers.Dense(latent_dim, activation='relu')(x)
#     x = layers.Add()([x, y])

    
    x = layers.Dense(n_window, activation=None)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Reshape((n_window, 1))(x)
    
    x = layers.Conv1D(n_decode_filters,
                     kernel_size=3,
                     padding='same',
                     activation='relu')(x)
    
    y = layers.Conv1D(n_decode_filters,
                     kernel_size=3,
                     padding='same',
                     activation='relu')(x)
    y = layers.Conv1D(n_decode_filters,
                     kernel_size=3,
                     padding='same',
                     activation='relu')(y)
    x = layers.Add()([x, y])
    x = layers.Dropout(0.5)(x)
    
    y = layers.Conv1D(n_decode_filters,
                     kernel_size=3,
                     padding='same',
                     activation='relu')(x)
    y = layers.Conv1D(n_decode_filters,
                     kernel_size=3,
                     padding='same',
                     activation='relu')(y)
    x = layers.Add()([x, y])
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv1D(n_ch, 1, padding="same", activation=None)(x)

    x_out = x
    decoder_net = Model(z_in, x_out, name='decoder')
    return decoder_net




#     y = layers.Dense(latent_dim, activation='tanh')(x)
#     y = layers.Dense(latent_dim, activation='tanh')(y)
#     x = layers.Add()([x, y])
    
#     y = layers.Dense(latent_dim, activation='tanh')(x)
#     y = layers.Dense(latent_dim, activation='tanh')(y)
#     x = layers.Add()([x, y])
    
#     y = layers.Dense(latent_dim, activation='tanh')(x)
#     y = layers.Dense(latent_dim, activation='tanh')(y)
#     x = layers.Add()([x, y])
    
#     y = layers.Dense(latent_dim, activation='tanh')(x)
#     y = layers.Dense(latent_dim, activation='tanh')(y)
#     x = layers.Add()([x, y])
    
#     x = layers.Dense(n_window//16, activation="relu")(x)
#     x = layers.Dense(n_window//8, activation="relu")(x)
#     x = layers.Dense(n_window//4, activation="relu")(x)
#     x = layers.Dense(n_window//2, activation="relu")(x)
#     x = layers.Dense(n_window, activation="relu")(x)
    
#     x = layers.Dense(n_window*n_ch, activation=None)(x)
    
#     x = layers.Reshape((n_window, n_ch))(x)