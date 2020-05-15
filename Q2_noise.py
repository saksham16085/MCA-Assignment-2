import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import os
import pickle as pkl
from skimage import util
import random

def get_label(argument):
    switcher = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
    }
    return switcher.get(argument,9)
noises = os.listdir('./_background_noise_')
folders = ['training','validation']
for i in folders:
    path = './'+i
    X = []
    Y = []

    for F in os.listdir(path):
        print("************************")
        print(F)
        print("************************")
        folder = path + '/' + F
        label_ = get_label(F)
        for file in os.listdir(folder):
            filename = folder+'/'+file

            sample_rate, signal = scipy.io.wavfile.read(filename)  # File assumed to be in the same directory
            flag = random.choice([1, 2])
            if flag == 2:
                noise_file = random.choice([0, 1, 2, 3, 4, 5])
                rate_noise, audio_noise = scipy.io.wavfile.read('./_background_noise_/' + noises[noise_file])
                audio_noise = audio_noise[:len(signal)]
                signal = signal + 0.3*audio_noise
            emphasized_signal = np.append(signal[0], signal[1:] - 0.95 * signal[:-1])
            audio_ = np.zeros((16000))
            audio_[:len(emphasized_signal)] = emphasized_signal
            emphasized_signal = audio_
            frame_length = 256
            frame_step = 172
            frames = util.view_as_windows(emphasized_signal, window_shape=(frame_length,), step=frame_step)
            frames *= np.hanning(frame_length)
            NFFT = 512
            p_spectrum = ((1.0 / NFFT) * ((np.abs(np.fft.rfft(frames, NFFT))) ** 2))
            nfilt = 80
            freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
            mel = np.linspace(0, freq_mel, nfilt + 2)
            hz = (700 * (10 ** (mel / 2595) - 1))
            bin = np.floor((NFFT + 1) * hz / sample_rate)
            filter_bank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
            for filter_iter in range(1, nfilt + 1):
                lower_val = int(bin[filter_iter - 1])
                equal = int(bin[filter_iter])
                more_val = int(bin[filter_iter + 1])

                for k in range(lower_val, equal):
                    filter_bank[filter_iter - 1, k] = (k - bin[filter_iter - 1]) / (
                                bin[filter_iter] - bin[filter_iter - 1])
                for k in range(equal, more_val):
                    filter_bank[filter_iter - 1, k] = (bin[filter_iter + 1] - k) / (
                                bin[filter_iter + 1] - bin[filter_iter])
            filter_banks = np.dot(p_spectrum, filter_bank.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
            filter_banks = 20 * np.log10(filter_banks)
            mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
            cep_lifter = 22
            (nframes, ncoeff) = mfcc.shape
            n = np.arange(ncoeff)
            sin_filter = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
            mfcc *= sin_filter
            X.append(mfcc)
            Y.append(label_)
    print(len(X),len(Y))
    pkl.dump(X, open(i+'_mfcc_features_noise_final', 'wb'))
    pkl.dump(Y, open(i+'_mfcc_labels_noise_final', 'wb'))