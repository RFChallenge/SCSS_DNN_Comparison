import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
from tqdm import tqdm
import pickle
from tqdm import tqdm

import rfcutils.qpsk_helper_fn as qpskfn
import rfcutils.ofdm_helper_fn as ofdmfn


import random

sig_len = 1280

soi_type = 'QPSK'
interference_sig_type = 'OFDM'

random.seed(0)
np.random.seed(0)

all_sig_mixture, all_sig1, all_sig2 = [], [] , []
all_val_sig_mixture, all_val_sig1, all_val_sig2 = [], [] , []

for idx in tqdm(range(10000)):
    for target_sinr in np.arange(-30,4.5,1.5):
        sig1, _, _, _ = qpskfn.generate_qpsk_signal(sig_len//16 + 80)
        sig2, _, _ = ofdmfn.generate_ofdm_signal(56*(sig_len//80 + 4))
        
        start_idx0 = np.random.randint(len(sig1)-sig_len)
        sig1 = sig1[start_idx0:start_idx0+sig_len]

        start_idx = np.random.randint(len(sig2)-sig_len)
        sig2 = sig2[start_idx:start_idx+sig_len]

        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        coeff *= np.exp(1j*2*np.pi*np.random.rand())
        
        noise = 0.1 * 1./np.sqrt(2) * (np.random.randn(sig_len) + 1j*np.random.randn(sig_len))
        sig_mixture = sig1 + sig2 * coeff + noise
        all_sig_mixture.append(sig_mixture)
        all_sig1.append(sig1)
        all_sig2.append(sig2*coeff)
        
#####
for idx in tqdm(range(500)):
    for target_sinr in np.arange(-30,4.5,1.5):
        sig1, _, _, _ = qpskfn.generate_qpsk_signal(sig_len//16 + 80)
        sig2, _, _ = ofdmfn.generate_ofdm_signal(56*(sig_len//80 + 4))
        
        start_idx0 = np.random.randint(len(sig1)-sig_len)
        sig1 = sig1[start_idx0:start_idx0+sig_len]

        start_idx = np.random.randint(len(sig2)-sig_len)
        sig2 = sig2[start_idx:start_idx+sig_len]

        coeff = np.sqrt(np.mean(np.abs(sig1)**2)/(np.mean(np.abs(sig2)**2)*(10**(target_sinr/10))))
        coeff *= np.exp(1j*2*np.pi*np.random.rand())
        
        noise = 0.1 * 1./np.sqrt(2) * (np.random.randn(sig_len) + 1j*np.random.randn(sig_len))
        sig_mixture = sig1 + sig2 * coeff + noise
        all_val_sig_mixture.append(sig_mixture)
        all_val_sig1.append(sig1)
        all_val_sig2.append(sig2*coeff)
        
all_sig_mixture = np.array(all_sig_mixture)
all_sig1 = np.array(all_sig1)
all_sig2 = np.array(all_sig2)

all_val_sig_mixture = np.array(all_val_sig_mixture)
all_val_sig1 = np.array(all_val_sig1)
all_val_sig2 = np.array(all_val_sig2)

window_len = sig_len

sig1_out = all_sig1.reshape(-1,window_len)
out1_comp = np.dstack((sig1_out.real, sig1_out.imag))

sig_mix_out = all_sig_mixture.reshape(-1,window_len)
mixture_bands_comp = np.dstack((sig_mix_out.real, sig_mix_out.imag))

sig1_val_out = all_val_sig1.reshape(-1,window_len)
out1_val_comp = np.dstack((sig1_val_out.real, sig1_val_out.imag))

sig_mix_val_out = all_val_sig_mixture.reshape(-1,window_len)
mixture_bands_val_comp = np.dstack((sig_mix_val_out.real, sig_mix_val_out.imag))


pickle.dump((mixture_bands_comp, out1_comp, mixture_bands_val_comp, out1_val_comp), open(os.path.join('dataset',f'{soi_type}_{interference_sig_type}_sigsep_dataset_large.pickle'), 'wb'), protocol=4)

