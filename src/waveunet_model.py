import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_wunet_model(input_shape, k_sz=15, merge_k_sz=5, lr=0.0003, k_neurons=32):
    n_window = input_shape[0]
    n_ch = 2

    in0 = layers.Input(shape=input_shape)
    x = in0
        
    upsamp_blocks = []
    for n_layer, k in enumerate([1, 2, 3, 4, 5]):
        conv = layers.Conv1D(k * k_neurons, k_sz, activation=None, padding="same")(x)
        conv = tf.keras.layers.LeakyReLU()(conv)
        upsamp_blocks.append(conv)
        pool = tf.keras.layers.Lambda(lambda x: x[:,::2,:])(conv)        
        x = pool
    
    convm = tf.keras.layers.Conv1D(6 * k_neurons, k_sz, padding="same")(x)
    convm = tf.keras.layers.LeakyReLU()(convm)
    x = convm
    
    for n_layer, k in enumerate([5, 4, 3, 2, 1]):
        deconv = layers.Conv1DTranspose(k * k_neurons, merge_k_sz, strides=2, padding="same")(x)
        uconv = layers.concatenate([deconv, upsamp_blocks[-(n_layer+1)]])
        uconv = layers.Conv1D(k * k_neurons, merge_k_sz, padding="same")(uconv)
        uconv = tf.keras.layers.LeakyReLU()(uconv)
        x = uconv
    
    output_layer = layers.Conv1D(n_ch, 1, padding="same", activation=None)(x)
    x_out = output_layer
    supreg_net = Model(in0, x_out, name='supreg')
    
    supreg_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.losses.MeanSquaredError()])    

    return supreg_net