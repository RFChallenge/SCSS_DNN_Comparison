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
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers



def get_dnn_model(input_shape, lr=0.0003):

    input_window = input_shape[0]
    n_ch = input_shape[1]
    
    window_len = input_window
    
    in0 = layers.Input(shape=(input_window, n_ch))
    x = layers.Flatten()(in0)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(input_window*n_ch, activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    y = layers.Dense(input_window*n_ch, activation=None)(x)
    y = layers.Activation("relu")(y)
    y = layers.Dense(input_window*n_ch, activation=None)(y)
    y = layers.BatchNormalization()(y)
    x = layers.Add()([x, y])
    x = layers.Activation("relu")(x)

    y = layers.Dense(input_window*n_ch, activation=None)(x)
    y = layers.Activation("relu")(y)
    y = layers.Dense(input_window*n_ch, activation=None)(y)
    y = layers.BatchNormalization()(y)
    x = layers.Add()([x, y])
    x = layers.Activation("relu")(x)

    y = layers.Dense(input_window*n_ch, activation=None)(x)
    y = layers.Activation("relu")(y)
    y = layers.Dense(input_window*n_ch, activation=None)(y)
    y = layers.BatchNormalization()(y)
    x = layers.Add()([x, y])
    x = layers.Activation("relu")(x)
    
    x_out = layers.Dense(window_len*n_ch, activation=None)(x)
    x_out = layers.Reshape((window_len, n_ch))(x_out)
    
    dnn = Model(in0, x_out, name='dnn')

    dnn.compile(optimizer=k.optimizers.Adam(learning_rate=lr),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.losses.MeanSquaredError()])
    return dnn
