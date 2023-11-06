from dataset.SourceDataset import SourceDataset
import os
from tqdm import tqdm
import random


class MaestroDataset(SourceDataset):
    def __init__(self, root_dir='../maestro-v3.0.0', noise_dataset=None):
        self.root_dir = root_dir
        self.samples = self.read_all_file()
        self.noise_dataset = noise_dataset

    
    def read_all_file(self):
        data = []
        # iterate and only check directories
        for noise in tqdm(os.listdir(self.root_dir), desc='Reading From Maestro'):
            noise_dir = os.path.join(self.root_dir, noise)
            if not os.path.isdir(noise_dir):
                continue
            noise_files = os.listdir(noise_dir)
            noise_files = [f for f in noise_files if f.endswith('.wav')]
            random.shuffle(noise_files)
            noise_files = noise_files[:10]
            for f in noise_files:
                wav_data = self.read_wav(os.path.join(noise_dir, f), amplification_factor=0.2)
                data.append(wav_data)
        return data

