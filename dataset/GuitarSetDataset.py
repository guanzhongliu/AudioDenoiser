from dataset.SourceDataset import SourceDataset
import os
from tqdm import tqdm


class GuitarSetDataset(SourceDataset):
    def __init__(self, noise_dataset=None):
        self.root_dir = "../data/GuitarSet-audio_mono-mic"
        self.samples = self.read_all_file()
        self.noise_dataset = noise_dataset

    
    def read_all_file(self):
        # iterate and only read wav files
        data = []
        for f in tqdm(os.listdir(self.root_dir), desc='Reading From GuitarSet'):
            if not f.endswith('.wav'):
                continue
            data.append(os.path.join(self.root_dir, f))
        return data

