import numpy as np
from scipy.io import wavfile
from scipy.linalg import toeplitz
import math
from scipy.fftpack import fft

"""
This script is modified for the task of noise reduction in musical instruments. 
It computes metrics suitable for evaluating the performance of noise reduction algorithms in non-speech audio.

Usage: 
    wss_dist, llr_mean, snr_mean, segSNR = compute_metrics(cleanFile, enhancedFile, Fs, path)
    cleanFile: clean audio as array or path if path is equal to 1
    enhancedFile: enhanced audio as array or path if path is equal to 1
    Fs: sampling rate, usually equals to 8000 or 16000 Hz
    path: whether the "cleanFile" and "enhancedFile" arguments are in .wav format or in numpy array format, 
          1 indicates "in .wav format"
          
Example call:
    wss_output, llr_output, snr_output, segsnr_output = \
            compute_metrics(target_audio, output_audio, 16000, 0)
"""

def compute_metrics(cleanFile, enhancedFile, Fs, path):
    alpha = 0.95

    if path == 1:
        sampling_rate1, data1 = wavfile.read(cleanFile)
        sampling_rate2, data2 = wavfile.read(enhancedFile)
        if sampling_rate1 != sampling_rate2:
            raise ValueError("The two files do not match!\n")
    else:
        data1 = cleanFile
        data2 = enhancedFile
        sampling_rate1 = Fs
        sampling_rate2 = Fs

    if len(data1) != len(data2):
        length = min(len(data1), len(data2))
        data1 = data1[0:length] + np.spacing(1)
        data2 = data2[0:length] + np.spacing(1)

    # compute the WSS measure
    wss_dist_vec = wss(data1, data2, sampling_rate1)
    wss_dist_vec = np.sort(wss_dist_vec)
    wss_dist = np.mean(wss_dist_vec[0 : round(np.size(wss_dist_vec) * alpha)])

    # compute the LLR measure
    LLR_dist = llr(data1, data2, sampling_rate1)
    LLRs = np.sort(LLR_dist)
    LLR_len = round(np.size(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[0:LLR_len])

    # compute the SNRseg
    snr_dist, segsnr_dist = snr(data1, data2, sampling_rate1)
    snr_mean = snr_dist
    segSNR = np.mean(segsnr_dist)

    return wss_dist, llr_mean, snr_mean, segSNR

# Place the definitions of the functions wss, llr, lpcoeff, snr here
# (As they were in your original code)

# Example of usage
# wss_output, llr_output, snr_output, segsnr_output = compute_metrics(target_audio, output_audio, 16000, 0)
