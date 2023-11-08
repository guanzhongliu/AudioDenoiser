from typing import List
from dataset.SourceDataset import SourceDataset
import random

class MixSourceDataset(SourceDataset):
    def __init__(self, datasets: List[SourceDataset]):
        self.datasets = datasets

    def __len__(self):
        l = 0
        for dataset in self.datasets:
            l += len(dataset)
        return l
    
    def __getitem__(self, index) -> [tuple, tuple]:
        # randomly choose a dataset
        dataset = self.datasets[index % len(self.datasets)]

        sample, noise = None, None
        while sample is None or noise is None:
            (sample, sample_rate), (noise, noise_rate) = dataset.get_random_sample()
            if sample is None or noise is None:
                print("Got a empty sample!")
        return (sample, sample_rate), (noise, noise_rate)

    def get_random_sample(self):
        return self.__getitem__(random.randint(0, len(self)))

