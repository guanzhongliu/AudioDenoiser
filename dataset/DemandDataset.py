import torch
from dataset.WavDataset import WavDataset
import os
import librosa
from tqdm import tqdm


class DemandDataset(WavDataset):
    def __init__(self, config):
        
        self.root_dir = config.conf.dset.demand_dir
        self.structure_data = self.read_all_noise()
        self.samples = []
        for noise in self.structure_data:
            self.samples.extend(self.structure_data[noise])
        self.window_length = config.conf.seg_len_s_train
    
    def read_all_noise(self):
        data = {}
        # iterate and only check directories
        for noise in tqdm(os.listdir(self.root_dir), desc='Reading From DEMAND'):
            noise_dir = os.path.join(self.root_dir, noise)
            if not os.path.isdir(noise_dir):
                continue
            noise_files = os.listdir(noise_dir)
            data[noise] = []
            for file in noise_files:
                if not file.endswith('.wav'):
                    continue
                wav_data = self.read_wav(os.path.join(noise_dir, file), amplification_factor=4.0)
                data[noise].append(wav_data)
        return data

    def __len__(self):
        return len(self.samples)
