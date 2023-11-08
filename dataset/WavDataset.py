import torch
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np

# Basic dataset class for reading and processing wav files
class WavDataset(Dataset):
    def __init__(self, config):
        self.samples = []
        self.window_length = config.conf.seg_len_s_train

    def read_wav(self, file_path, amplification_factor=1.0):
        waveform, sample_rate = sf.read(file_path)
        # stereo to mono
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        return waveform * amplification_factor, sample_rate
    
    def random_sample_from_wav(self, waveform, sample_rate):
        # Calculate the number of samples in the window length
        window_length_samples = self.window_length * sample_rate
        # Make sure the waveform is long enough
        if len(waveform) < window_length_samples:
            return None, None
        # Choose a random start point
        start_index = torch.randint(len(waveform) - window_length_samples, (1,)).item()
        # Extract the window from the waveform
        window = waveform[start_index : start_index + window_length_samples]

        return window, sample_rate

    def write_wav(self, file_path, waveform, sample_rate):
        sf.write(file_path, waveform, sample_rate)

    def do_stft(self, waveform, n_fft=2048, hop_length=512, win_length=2048):
        waveform = torch.from_numpy(waveform).float()
        window = torch.hann_window(win_length)
        stft = torch.stft(waveform, n_fft, hop_length, win_length, window, return_complex=False)
        return stft
    
    def __getitem__(self, index):
        sample, sample_rate = self.samples[index]

        return self.random_sample_from_wav(sample, sample_rate)
    
    def get_random_sample(self):
        index = torch.randint(len(self.samples), (1,)).item()
        return self.__getitem__(index)