from dataset.WavDataset import WavDataset
import os
from tqdm import tqdm


class DemandDataset(WavDataset):
    def __init__(self):
        self.root_dir = "../data/DEMAND"
        self.structure_data = self.read_all_noise()
        self.samples = []
        for noise in self.structure_data:
            self.samples.extend(self.structure_data[noise])
    
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
                data[noise].append(os.path.join(noise_dir, file))
        return data

    def __len__(self):
        return len(self.samples)
