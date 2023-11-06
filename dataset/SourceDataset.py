from dataset.WavDataset import WavDataset

class SourceDataset(WavDataset):
    def __init__(self, root_dir, noise_dataset=None):
        super().__init__()
        self.root_dir = root_dir
        self.noise_dataset = noise_dataset
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # if noise dataset is provided, get a random noise sample from it
        if self.noise_dataset is not None:
            noise_sample = self.noise_dataset.get_random_sample()
        else:
            noise_sample = None
        sample, sample_rate = self.samples[index]

        return self.random_sample_from_wav(sample, sample_rate), noise_sample