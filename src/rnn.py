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



def get_rnn_model(input_shape, n_gru=512, lr=0.0003):

    input_window = input_shape[0]
    n_ch = input_shape[1]

    in0 = layers.Input(shape=(input_window, n_ch))

    seq = layers.Bidirectional(layers.GRU(n_gru, return_sequences=True))(in0)
    seq = layers.Bidirectional(layers.GRU(n_gru//2, return_sequences=True))(seq)
    seq = layers.Bidirectional(layers.GRU(n_gru//4, return_sequences=True))(seq)

    x_out = layers.TimeDistributed(layers.Dense(n_ch, activation=None))(seq)
    
    rnn = Model(in0, x_out, name='rnn')

    rnn.compile(optimizer=k.optimizers.Adam(learning_rate=lr, amsgrad=True),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.losses.MeanSquaredError()])
    return rnn
