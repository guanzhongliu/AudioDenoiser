from dataset.SourceDataset import SourceDataset
import os
from tqdm import tqdm


class PhenicxDataset(SourceDataset):
    def __init__(self, noise_dataset=None):
        self.root_dir = "../PHENICX-Anechoic"
        self.samples = self.read_all_file()
        self.noise_dataset = noise_dataset
    
    def read_all_file(self):
        data = []
        # iterate and only check directories
        for noise in tqdm(os.listdir(self.root_dir), desc='Reading From Phenicx'):
            noise_dir = os.path.join(self.root_dir, noise)
            if not os.path.isdir(noise_dir):
                continue
            noise_files = os.listdir(noise_dir)
            for f in noise_files:
                if not f.endswith('.wav'):
                    continue
                wav_data = self.read_wav(os.path.join(noise_dir, f), amplification_factor=1.0)
                data.append(wav_data)
        return data
