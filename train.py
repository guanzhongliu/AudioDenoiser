from dataset import DemandDataset, GuitarSetDataset, PhenicxDataset, MaestroDataset
import librosa

demandDataset = DemandDataset(root_dir='../DEMAND')
guitarSetDataset = GuitarSetDataset(root_dir='../GuitarSet-audio_mono-mic', noise_dataset=demandDataset)
phenicxDataset = PhenicxDataset(root_dir='../PHENICX-Anechoic', noise_dataset=demandDataset)
maestroDataset = MaestroDataset(root_dir='../maestro-v3.0.0', noise_dataset=demandDataset)

(sample, sample_rate), (noise_sample, noise_sample_rate) = guitarSetDataset.get_random_sample()
if sample_rate != noise_sample_rate:
    noise_sample = librosa.resample(noise_sample, noise_sample_rate, sample_rate)
combined_sample = sample + noise_sample

# write combined sample to file
phenicxDataset.write_wav('combined_guitar_sample.wav', combined_sample, sample_rate)

(sample, sample_rate), (noise_sample, noise_sample_rate) = phenicxDataset.get_random_sample()
if sample_rate != noise_sample_rate:
    noise_sample = librosa.resample(noise_sample, noise_sample_rate, sample_rate)
combined_sample = sample + noise_sample

phenicxDataset.write_wav('combined_phenicx_sample.wav', combined_sample, sample_rate)


