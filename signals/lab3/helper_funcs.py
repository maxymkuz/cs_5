import numpy as np

import matplotlib

import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi']= 100
matplotlib.rcParams['figure.figsize'] = 15, 5
from math import pi

from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y

def plot_spectrum_stem(sig, sample_rate, lower=0, upper=None):
    yf = rfft(sig)
    xf = rfftfreq(len(sig), 1 / sample_rate)
    if upper is None:
        upper = len(xf)

    plt.stem(xf[lower:upper], np.abs(yf)[lower:upper])
    plt.xlabel('Freq, Hz')
    plt.ylabel('Values')
    plt.show()

def plot_sig_rfft(sig, sample_rate):
    yf = rfft(sig)
    xf = rfftfreq(len(sig), 1 / sample_rate)
    plt.plot(xf, np.abs(yf))
    plt.xlabel('Freq, Hz')
    plt.ylabel('Values')
    plt.show()

    # normalize the magnitude
    transformed_sig = abs(fft(sig) / len(sig))
    # plt.plot(timestamps, transformed_sig, color="red")
    # plt.show()
    return transformed_sig


