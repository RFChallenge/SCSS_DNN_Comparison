import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle
from tqdm import tqdm

import rfcutils
import rfcutils.ofdm_helper_fn as ofdmfn

from src import unet_model as unet

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

for long_k_sz in [101, 3, 201, 5, 11, 15, 65, 64, 63, 61, 81, 80, 71, 51]:
    def scheduler(epoch, lr=0.0003):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.04)
    lr_callback = LearningRateScheduler(scheduler)
    
    model_name = f'{soi_type}_{interference_sig_type}_{window_len}_K{long_k_sz}'
    earlystopping = EarlyStopping(monitor='val_loss', patience=100)
    checkpoint = ModelCheckpoint(filepath=f'models/unet/tmp/{model_name}/checkpoint', monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)    
    print(f'Training {model_name}')

    nn_model = unet.get_unet_model0((window_len, 2), k_sz=3, long_k_sz=long_k_sz, k_neurons=32, lr=0.0003)
    nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, shuffle=True, verbose=1, validation_data=(mixture_bands_val_comp, out1_val_comp), callbacks=[checkpoint, earlystopping, lr_callback])
    
    if long_k_sz == 3:
        def scheduler(epoch, lr=0.0003):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.04)
        lr_callback = LearningRateScheduler(scheduler)
    
        model_name = f'{soi_type}_{interference_sig_type}_{window_len}_K{long_k_sz}_2x'
        earlystopping = EarlyStopping(monitor='val_loss', patience=100)
        checkpoint = ModelCheckpoint(filepath=f'models/unet/tmp/{model_name}/checkpoint', monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)    
        print(f'Training {model_name}')
        nn_model = unet.get_unet_model0((window_len, 2), k_sz=3, long_k_sz=long_k_sz, k_neurons=64, lr=0.0003)
        nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, shuffle=True, verbose=1, validation_data=(mixture_bands_val_comp, out1_val_comp), callbacks=[checkpoint, earlystopping, lr_callback])