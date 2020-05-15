import os
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
import os
import pickle as pkl
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
        folder = path+'/'+F
        label_ = get_label(F)
        for file in os.listdir(folder):


            filename = folder+'/'+file


            rate, audio = wavfile.read(filename)
            N = audio.shape[0]
            flag = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            if flag <=5:
                noise_file = random.choice([0,1,2,3,4,5])
                rate_noise, audio_noise = wavfile.read('./_background_noise_/'+noises[noise_file])
                audio_noise = audio_noise[:N]
                audio = audio+0.3*audio_noise
            audio_ = np.zeros((16000))
            audio_[:len(audio)] = audio
            audio = audio_

            # print(f'Audio length: {L:.2f} seconds')
            #
            # f, ax = plt.subplots()
            # ax.plot(np.arange(N) / rate, audio)
            # ax.set_xlabel('Time [s]')
            # ax.set_ylabel('Amplitude [unknown]');
            # plt.show()

            M=256
            slices = util.view_as_windows(audio, window_shape=(M,), step=172)
            # print(f'Audio shape: {audio.shape}, Sliced audio shape: {slices.shape}')
            win = np.hanning(M + 1)[:-1]
            slices = slices * win
            spectrum = []
            # for a in slices:
            #     leng = len(a)
            #     mag = []
            #     ks = np.arange(0, leng, 1)
            #     for b in range(int(leng/2)):
            #         ex = (1j*2*np.pi*ks*b)/leng
            #         va = np.abs(np.sum(a*np.exp(ex))/leng)*2
            #         mag.append(va)
            #     spectrum.append(mag)

            spectrum = np.fft.fft(slices)
            n = np.asarray(spectrum).shape[1]
            spectrum = spectrum[:,:(n)//2+1]

            spectrum = np.array(np.abs(spectrum)).T
            n = spectrum.shape[1]
            spectrum[spectrum == 0] = np.finfo(np.float64).eps
            S = 10*np.log10(spectrum)

            # f, ax = plt.subplots(figsize=(4.8, 2.4))
            # ax.imshow(S, origin='lower', cmap='viridis',
            #           extent=(0, len(S), 0, rate / 2 / 1000))
            # ax.axis('tight')
            # ax.set_ylabel('Frequency [kHz]')
            # ax.set_xlabel('Time [s]');
            # plt.show()
            X.append(S)
            Y.append(label_)
    print(spectrum.shape)
    pkl.dump(X, open(i+'_spectograms_features_noise_final', 'wb'))
    pkl.dump(Y, open(i+'_spectograms_labels_noise_final', 'wb'))
    print(spectrum.shape)
    print(len(X))
    print(len(Y))
