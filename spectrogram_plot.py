from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from skimage import util
import os

def get_Hz_scale_vec(ks,sample_rate,Npoints):
    freq_Hz = ks*sample_rate/Npoints
    freq_Hz  = [float(i/1000) for i in freq_Hz ]
    freq_Hz_ = ["{:4.2f}".format(i) for i in freq_Hz]
    return(freq_Hz_ )

rate, audio = wavfile.read('./training/eight/0bde966a_nohash_1.wav')
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
    leng_arr = np.arange(0, leng, 1)
    for b in range(int(leng/2)):
        ex = (1j*2*np.pi*leng_arr*b)/leng
        va = np.abs(np.sum(a*np.exp(ex))/leng)*2
        mag.append(va)
    spectrum.append(mag)

spectrum = np.array(np.abs(spectrum)).T
n = spectrum.shape[1]
spectrum[spectrum == 0] = np.finfo(np.float64).eps
spectrum = 10*np.log10(spectrum)
plt.figure()
plt_spec = plt.imshow(spectrum, origin='lower')
yticks = 15

ks = np.linspace(0, spectrum.shape[0], yticks)
ksHz = get_Hz_scale_vec(ks, rate, 256)
plt.yticks(ks, ksHz)
plt.ylabel("Frequency (kHz)")
xticks = 5
ts_spec = np.linspace(0, spectrum.shape[1], xticks)

ts_spec_sec = ["{:2.2f}".format(i) for i in np.linspace(0, 1 * 15872 / 16000, xticks)]
plt.xticks(ts_spec,ts_spec_sec)
plt.xlabel("Time (sec)")
plt.title("Spectrogram")
plt.colorbar(None, use_gridspec=True)
plt.show()
