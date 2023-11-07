from dataset.SourceDataset import SourceDataset
import os
from tqdm import tqdm


class GuitarSetDataset(SourceDataset):
    def __init__(self, config, noise_dataset=None):
        self.root_dir = config.guitar_dir
        self.samples = self.read_all_file()
        self.noise_dataset = noise_dataset
        self.window_length = config.seg_len_s_train

    
    def read_all_file(self):
        # iterate and only read wav files
        data = []
        for f in tqdm(os.listdir(self.root_dir), desc='Reading From GuitarSet'):
            if not f.endswith('.wav'):
                continue
            wav_data = self.read_wav(os.path.join(self.root_dir, f), amplification_factor=1.0)
            data.append(wav_data)
        return data

