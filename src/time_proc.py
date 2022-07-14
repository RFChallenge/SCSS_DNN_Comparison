import os
import numpy as np
from scipy.linalg import hankel

def moving_window(sig, window_len=320, seg_len=128):
    sig_window_out = np.zeros((sig.shape[0], sig.shape[1], seg_len), dtype=complex)
    for iidx in range(sig.shape[0]):
        sig_mixture = np.hstack((np.zeros(seg_len//2), sig[iidx], np.zeros(seg_len-seg_len//2)))
        sig_hankel = hankel(sig_mixture[:-seg_len], sig_mixture[-seg_len:])
        sig_window_out[iidx,:,:] = sig_hankel
    return sig_window_out.reshape(-1, window_len, seg_len)


def long_window(sig, window_len=320, seg_len=128):
    all_windows = []
    for iidx in range(sig.shape[0]):
        sig_windows = []
        sig_mixture = np.hstack((np.zeros(seg_len//2), sig[iidx], np.zeros(seg_len-seg_len//2)))
        for n in range(len(sig_mixture)//window_len):
            start_idx = n*(window_len)
            end_idx = start_idx + window_len + seg_len
            sig_windows.append(sig_mixture[start_idx:end_idx])
        all_windows.extend(np.array(sig_windows))
    return np.array(all_windows)