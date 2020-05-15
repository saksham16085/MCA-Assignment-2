import scipy
import numpy as np
from skimage import util
from matplotlib import pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct

sample_rate, signal = scipy.io.wavfile.read('./training/eight/00b01445_nohash_0.wav')  # File assumed to be in the same directory
emphasized_signal = np.append(signal[0], signal[1:] - 0.95 * signal[:-1])
audio_ = np.zeros((16000))
audio_[:len(emphasized_signal)] = emphasized_signal
emphasized_signal = audio_
frame_length = 256
frame_step = 172
frames = util.view_as_windows(emphasized_signal, window_shape=(frame_length,), step=frame_step)
frames *= np.hanning(frame_length)
NFFT = 512
p_spectrum = ((1.0/NFFT)*((np.abs(np.fft.rfft(frames, NFFT)))**2))
nfilt= 80
freq_mel = (2595*np.log10(1+(sample_rate/2)/700))
mel = np.linspace(0,freq_mel, nfilt + 2)
hz = (700*(10**(mel/2595)-1))
bin = np.floor((NFFT+1)*hz/sample_rate)
filter_bank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for filter_iter in range(1, nfilt + 1):
    lower_val = int(bin[filter_iter - 1])
    equal = int(bin[filter_iter])
    more_val = int(bin[filter_iter + 1])

    for k in range(lower_val, equal):
        filter_bank[filter_iter - 1, k] = (k - bin[filter_iter - 1]) / (bin[filter_iter] - bin[filter_iter - 1])
    for k in range(equal, more_val):
        filter_bank[filter_iter - 1, k] = (bin[filter_iter + 1] - k) / (bin[filter_iter + 1] - bin[filter_iter])
filter_banks = np.dot(p_spectrum, filter_bank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20*np.log10(filter_banks)
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
cep_lifter = 22
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
sin_filter = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= sin_filter


plt.figure()
plt_spec = plt.imshow(mfcc, origin='lower')

plt.title("MFCC Plot")
plt.colorbar(None, use_gridspec=True)
plt.show()