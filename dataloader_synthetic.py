from typing import Dict, List
import numpy as np
import dataloader
import librosa
import os

UNIFORM_SAMPLE_RATE = dataloader.UNIFORM_SAMPLE_RATE

def get_dataset_synthetic(path: str) -> Dict[str, List[np.ndarray]]:
    res = {}
    for cls in os.listdir(path):
        res[cls] = []
        for file in os.listdir(f'{path}/{cls}'):
            if not file.endswith('.wav'): continue

            orig_samples, orig_sr = librosa.load(f'{path}/{cls}/{file}', sr = None)
            new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = dataloader.UNIFORM_SAMPLE_RATE)
            assert new_samples.shape == (round(dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS),), new_samples.shape
            res[cls].append(new_samples)
    return res

if __name__ == '__main__':
    d = get_dataset_synthetic('dataset-synthetic/')
    print({ k: len(v) for k, v in d.items() })
