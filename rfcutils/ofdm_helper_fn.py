import numpy as np

from scipy.signal import convolve
from commpy.modulation import PSKModem, QAMModem

# based on ofdm_tx and ofdm_rx from https://github.com/veeresht/CommPy/blob/master/commpy/modulation.py, but fixed parameters to be ints to be compatible with latest numpy library
def ofdm_tx(x, nfft, nsc, cp_length):
    """ OFDM Transmit signal generation """
    assert nsc%2 == 0, f'nsc is not even: {nsc}'
    ofdm_tx_signal = np.array([])
    for i in range(0, x.shape[1]):
        symbols = x[:, i]
        ofdm_sym_freq = np.zeros(nfft, dtype=complex)
        ofdm_sym_freq[1:(nsc//2) + 1] = symbols[nsc//2:]
        ofdm_sym_freq[-nsc//2:] = symbols[0:nsc//2]
        ofdm_sym_time = np.fft.ifft(ofdm_sym_freq) * np.sqrt(nfft)
        if cp_length > 0:
            cp = ofdm_sym_time[-cp_length:]
            ofdm_tx_signal = np.concatenate((ofdm_tx_signal, cp, ofdm_sym_time))
        else:
            ofdm_tx_signal = np.concatenate((ofdm_tx_signal, ofdm_sym_time))
    return ofdm_tx_signal

def ofdm_rx(y, nfft, nsc, cp_length):
    """ OFDM Receive Signal Processing """
    num_ofdm_symbols = int(len(y) / (nfft + cp_length))
    x_hat = np.zeros([nsc, num_ofdm_symbols], dtype=complex)
    for i in range(0, num_ofdm_symbols):
        ofdm_symbol = y[i * nfft + (i + 1) * cp_length:(i + 1) * (nfft + cp_length)]
        symbols_freq = np.fft.fft(ofdm_symbol) / np.sqrt(nfft)
        x_hat[:, i] = np.concatenate((symbols_freq[-nsc//2:], symbols_freq[1:(nsc//2) + 1]))
    return x_hat



fft_len = 64
cp_len = 16
nsc = 56

mod = QAMModem(16)
# mod = PSKModem(4)

normalizing_coeff = 1

default_n_sym = nsc*1000
def generate_ofdm_signal(n_sym=default_n_sym, coeff=None, mod=mod, nfft=fft_len, cp=cp_len, nsc=nsc):
    assert n_sym%nsc == 0, f'Unable to reshape {n_sym} symbols to fit {nsc} subcarriers'
    if coeff is None:
        coeff = normalizing_coeff
    sB = np.random.randint(2, size=(int(mod.num_bits_symbol*n_sym)))
    sQ = mod.modulate(sB)
    sQ = sQ.reshape(-1, nsc).T
    
    y_ofdm = ofdm_tx(sQ, nfft, nsc, cp)
    y_ofdm *= coeff
    return y_ofdm.astype(np.complex64), sQ, sB

def ofdm_demod(sig, coeff=None, mod=mod, nfft=fft_len, cp=cp_len, nsc=nsc):
    if coeff is None:
        coeff = normalizing_coeff
    x_hat = ofdm_rx(sig, nfft, nsc, cp)
    x_hat = x_hat/coeff
    sQ_est = x_hat
    bit_est = mod.demodulate(x_hat.T.flatten(), demod_type='hard')
#     noise_var = 0.1
#     bit_llr = mod.demodulate(x_hat.flatten()/coeff, demod_type='soft', noise_var=noise_var)
#     bit_prob = 1/(1+np.exp(bit_llr))
#     bit_prob = np.clip(bit_prob, 1e-12, 1-1e-12)
    return bit_est, sQ_est


def modulate_ofdm_signal(info_bits, coeff=None, mod=mod, nfft=fft_len, cp=cp_len, nsc=nsc):
    if coeff is None:
        coeff = normalizing_coeff
        
    sB = info_bits
    sQ = mod.modulate(sB)
    sQ = sQ.reshape(-1, nsc).T
    
    y_ofdm = ofdm_tx(sQ, nfft, nsc, cp)
    y_ofdm *= coeff
    return y_ofdm
