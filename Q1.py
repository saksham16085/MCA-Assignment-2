from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
import os
import pickle as pkl

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

folders = ['training','validation']
for i in folders:
    path = './'+i
    X = []
    Y = []
    for F in os.listdir(path):

        print("************************")
        print(F)
        print("************************")
        folder = path+'/'+F
        label_ = get_label(F)
        for file in os.listdir(folder):
            filename = folder+'/'+file


            rate, audio = wavfile.read(filename)
            N = audio.shape[0]
            audio_ = np.zeros((16000))
            audio_[:len(audio)] = audio
            audio = audio_
            M=256
            slices = util.view_as_windows(audio, window_shape=(M,), step=172)
            win = np.hanning(M + 1)[:-1]
            slices = slices * win
            spectrum = []
            for a in slices:
                leng = len(a)
                mag = []
                ks = np.arange(0, leng, 1)
                for b in range(int(leng/2)):
                    ex = (1j*2*np.pi*ks*b)/leng
                    va = np.abs(np.sum(a*np.exp(ex))/leng)*2
                    mag.append(va)
                spectrum.append(mag)

            spectrum = np.array(np.abs(spectrum)).T
            n = spectrum.shape[1]
            spectrum[spectrum == 0] = np.finfo(np.float64).eps
            S = 10*np.log10(spectrum)
            X.append(S)
            Y.append(label_)
    print(spectrum.shape)
    pkl.dump(X, open(i+'_spectograms_features_final', 'wb'))
    pkl.dump(Y, open(i+'_spectograms_labels_final', 'wb'))
    print(spectrum.shape)
    print(len(X))
    print(len(Y))
