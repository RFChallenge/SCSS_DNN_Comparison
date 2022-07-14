import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle
from tqdm import tqdm

import rfcutils
import rfcutils.ofdm_helper_fn as ofdmfn

from src.vae import get_encoder_model, get_decoder_model, get_vae_model

import tensorflow as tf
import random
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

sig_len = 1280

soi_type = 'QPSK'
interference_sig_type = 'OFDM'

random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


window_len = sig_len
mixture_bands_comp, out1_comp, mixture_bands_val_comp, out1_val_comp = pickle.load(open(os.path.join('dataset',f'{soi_type}_{interference_sig_type}_sigsep_dataset_large.pickle'), 'rb'))

mixture_bands_comp = mixture_bands_comp.reshape(-1, window_len, 2)
out1_comp = out1_comp.reshape(-1, window_len, 2)
mixture_bands_val_comp = mixture_bands_val_comp.reshape(-1, window_len, 2)
out1_val_comp = out1_val_comp.reshape(-1, window_len, 2)


model_name = f'{soi_type}_{interference_sig_type}_{window_len}'
earlystopping = EarlyStopping(monitor='val_mse', patience=100)
checkpoint = ModelCheckpoint(filepath=f'models/vae_longksz/tmp/{model_name}/checkpoint', monitor='val_mse', verbose=0, save_best_only=True, mode='min', save_weights_only=True)    
print(f'Training {model_name}')


latent_dim = sig_len//4
encoder_net = get_encoder_model(latent_dim, (window_len, 2), 32, first_kernel_size=101)
decoder_net = get_decoder_model(latent_dim, (window_len, 2), 32)
nn_model = get_vae_model(encoder_net, decoder_net, (window_len, 2), latent_dim, lr=0.0003)

nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, shuffle=True, verbose=1, validation_data=(mixture_bands_val_comp, out1_val_comp), callbacks=[checkpoint, earlystopping])