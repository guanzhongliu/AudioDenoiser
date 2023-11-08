from dataset import *
import hydra
from omegaconf import DictConfig
import logging
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)

@hydra.main(config_name="conf/conf.yaml")
def create_dataset(cfg: DictConfig):
    # logger.info current path
    demandDataset = DemandDataset(config=cfg)
    guitarSetDataset = GuitarSetDataset(config=cfg, noise_dataset=demandDataset)
    phenicxDataset = PhenicxDataset(config=cfg, noise_dataset=demandDataset)
    maestroDataset = MaestroDataset(config=cfg, noise_dataset=demandDataset)

    source_datasets = [guitarSetDataset, phenicxDataset, maestroDataset]
    mix_dataset = MixSourceDataset(source_datasets)

    # logger length
    logger.info("length of GuitarSetDataset: %s", len(guitarSetDataset))
    logger.info("length of PhenicxDataset: %s", len(phenicxDataset))
    logger.info("length of MaestroDataset: %s", len(maestroDataset))
    logger.info("length of MixSourceDataset: %s", len(mix_dataset))

    return mix_dataset

def generate_samples(dir_path, num, dataset):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # create clean dir and noisy dir
    clean_dir = os.path.join(dir_path, "clean")
    noisy_dir = os.path.join(dir_path, "noisy")
    for i in tqdm(range(num), desc="Generating samples in " + dir_path):
        (sample, sample_rate), (noise_sample, noise_sample_rate) = dataset.get_random_sample()
        dataset.write_wav(os.path.join(clean_dir, f'sample_{i}.wav'), sample, sample_rate)
        dataset.write_wav(os.path.join(noisy_dir, f'sample_{i}.wav'), dataset.mix_samples(sample, sample_rate, noise_sample, noise_sample_rate), sample_rate)



if __name__ == "__main__":
    dataset = create_dataset()
    generate_samples("../../../data/train", 100, dataset=dataset)


