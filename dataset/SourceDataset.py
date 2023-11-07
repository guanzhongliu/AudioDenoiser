from dataset.WavDataset import WavDataset
import librosa

class SourceDataset(WavDataset):
    def __init__(self, config, noise_dataset=None):
        super().__init__()
        self.noise_dataset = noise_dataset
        self.window_length = config.dataset.window_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # if noise dataset is provided, get a random noise sample from it
        if self.noise_dataset is not None:
            noise_sample = self.noise_dataset.get_random_sample()
        else:
            noise_sample = (None, None)
        sample, sample_rate = self.samples[index]

        return self.random_sample_from_wav(sample, sample_rate), noise_sample
    
    def mix_samples(self, sample, sample_rate, noise_sample, noise_sample_rate):
        if sample_rate != noise_sample_rate:
            noise_sample = librosa.resample(noise_sample, noise_sample_rate, sample_rate)
        # add the noise to the sample
        combined_sample = sample + noise_sample
        return combined_sample