import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle
from tqdm import tqdm

import rfcutils
import rfcutils.ofdm_helper_fn as ofdmfn

from src.densenn import get_dnn_model

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


model_name = f'{soi_type}_{interference_sig_type}_{window_len}'
earlystopping = EarlyStopping(monitor='val_loss', patience=100)
checkpoint = ModelCheckpoint(filepath=f'models/dnn/tmp/{model_name}/checkpoint', monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)    
print(f'Training {model_name}')

nn_model = get_dnn_model((window_len, 2), lr=0.0003)
    
nn_model.fit(mixture_bands_comp, out1_comp, epochs=2000, batch_size=32, shuffle=True, verbose=1, validation_data=(mixture_bands_val_comp, out1_val_comp), callbacks=[checkpoint, earlystopping])