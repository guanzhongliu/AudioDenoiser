from dataset import *
import hydra
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf/conf.yaml")
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

    # Get a random sample from the mix dataset
    # for i in range(5):
    #     (sample, sample_rate), (noise_sample, noise_sample_rate) = mix_dataset.get_random_sample()
    #     logger.info("sample: %s", sample)
    #     logger.info("sample_rate: %s", sample_rate)
    #     logger.info("noise_sample: %s", noise_sample)
    #     logger.info("noise_sample_rate: %s", noise_sample_rate)

    #     mix_dataset.write_wav(f'sample_combine_{i}.wav', mix_dataset.mix_samples(sample, sample_rate, noise_sample, noise_sample_rate), sample_rate)
    #     mix_dataset.write_wav(f'sample_{i}.wav', sample, sample_rate)
    #     mix_dataset.write_wav(f'sample_noise_{i}.wav', noise_sample, noise_sample_rate)

    return mix_dataset

if __name__ == "__main__":
    create_dataset()


