from dataset.WavDataset import WavDataset
import librosa
import numpy as np

class SourceDataset(WavDataset):
    def __init__(self, noise_dataset=None):
        super().__init__()
        self.noise_dataset = noise_dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # if noise dataset is provided, get a random noise sample from it
        if self.noise_dataset is not None:
            noise_sample = self.noise_dataset.get_random_sample()
        else:
            noise_sample = (None, None)
        sample, sample_rate = self.read_wav(self.samples[index])

        return self.random_sample_from_wav(sample, sample_rate), noise_sample
    
    def mix_samples(self, sample, sample_rate, noise_sample, noise_sample_rate, noise_factor=0.5):
        # if noise sample is not provided, return the original sample
        if sample_rate != noise_sample_rate:
            noise_sample = librosa.resample(noise_sample, noise_sample_rate, sample_rate)

        # calculate the RMS of the sample and noise 
        rms_sample = np.sqrt(np.mean(sample**2))
        rms_noise = np.sqrt(np.mean(noise_sample**2))

        # calculate the desired RMS of the noise
        desired_noise_rms = rms_sample * noise_factor

        # calculate the gain factor to apply to the noise
        gain_factor = desired_noise_rms / rms_noise

        # adjust the noise to match the desired RMS
        noise_sample_adjusted = noise_sample * gain_factor

        # combine the sample and noise
        combined_sample = sample + noise_sample_adjusted

        # clip the combined sample to the range of the data type, avoiding clipping
        max_int_value = np.iinfo(sample.dtype).max if np.issubdtype(sample.dtype, np.integer) else 1.0
        combined_sample = np.clip(combined_sample, -max_int_value, max_int_value)

        return combined_sample