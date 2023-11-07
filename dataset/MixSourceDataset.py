from torch.utils.data import Dataset
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
        
        return dataset.get_random_sample()

    def get_random_sample(self):
        return self.__getitem__(random.randint(0, len(self)))

