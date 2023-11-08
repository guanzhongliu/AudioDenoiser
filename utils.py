from dataset import *
import os
from tqdm import tqdm
import torch
import torch.nn as nn


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def power_compress(x):
    real = x[..., 0]
    imag = x[..., 1]
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

def create_dataset():
    # logger.info current path
    demandDataset = DemandDataset()
    guitarSetDataset = GuitarSetDataset(noise_dataset=demandDataset)
    phenicxDataset = PhenicxDataset(noise_dataset=demandDataset)
    maestroDataset = MaestroDataset(noise_dataset=demandDataset)

    source_datasets = [guitarSetDataset, phenicxDataset, maestroDataset]
    mix_dataset = MixSourceDataset(source_datasets)

    # logger length
    print("length of GuitarSetDataset: %s", len(guitarSetDataset))
    print("length of PhenicxDataset: %s", len(phenicxDataset))
    print("length of MaestroDataset: %s", len(maestroDataset))
    print("length of MixSourceDataset: %s", len(mix_dataset))

    return mix_dataset

def generate_samples(dir_path, num, dataset):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # create clean dir and noisy dir
    clean_dir = os.path.join(dir_path, "clean")
    noisy_dir = os.path.join(dir_path, "noisy")
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir, exist_ok=True)
    if not os.path.exists(noisy_dir):
        os.makedirs(noisy_dir, exist_ok=True)
    for i in tqdm(range(num), desc="Generating samples in " + dir_path):
        (sample, sample_rate), (noise_sample, noise_sample_rate) = dataset.get_random_sample()
        dataset.write_wav(os.path.join(clean_dir, f'sample_{i}.wav'), sample, sample_rate)
        dataset.write_wav(os.path.join(noisy_dir, f'sample_{i}.wav'), dataset.mix_samples(sample, sample_rate, noise_sample, noise_sample_rate), sample_rate)



if __name__ == "__main__":
    dataset = create_dataset()
    generate_samples("genData/train", 100, dataset=dataset)


