import numpy as np
import argparse
import random
import librosa
import os
import sys
import shutil
import torch
import soundfile

import dataloader
import mfcc
import mfcc_vae_1 as vae

FILTER_TYPE = 'ClusterFilterAug3'
FILTER_THRESH = 0.25
MAX_CLUSTERS = 256
MAX_WEIGHT = 4
VOTE_THRESH = 0.2

class DeploymentDataloader:
    def __init__(self, path: str):
        self.__files = []
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith('.wav'):
                    self.__files.append(f'{dirpath}/{filename}')
        self.__files.sort() # order files by deployment and then chronologically

    def __iter__(self):
        target_samples = dataloader.SAMPLE_DURATION_SECS * dataloader.UNIFORM_SAMPLE_RATE
        for file in self.__files:
            try:
                orig_samples, orig_sr = librosa.load(file, sr = None)
                new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = dataloader.UNIFORM_SAMPLE_RATE)
            except:
                print(f'failed to read file \'{file}\'... skipping...')

            clip_count = new_samples.shape[0] // target_samples
            front_split = random.randrange(new_samples.shape[0] % target_samples)
            trimmed = new_samples[front_split : front_split + clip_count * target_samples]
            assert trimmed.shape[0] // target_samples == clip_count and trimmed.shape[0] % target_samples == 0

            yield trimmed, [trimmed[i * target_samples : (i + 1) * target_samples] for i in range(clip_count)]

class ClusterFilterAug3:
    def __init__(self, max_clusters, max_weight, embedding_size, thresh):
        self.means = np.zeros((max_clusters, embedding_size))
        self.weights = np.zeros((max_clusters,))
        self.max_weight = max_weight
        self.max_clusters = max_clusters
        self.base_radius = thresh
    def insert(self, mean, std):
        assert mean.shape == self.means.shape[1:] and mean.shape == std.shape and self.means.shape[0] == self.max_clusters and self.weights.shape == (self.max_clusters,)
        l2_norm = np.sqrt(np.sum((self.means - mean) ** 2, axis = 1))
        close = l2_norm <= np.sqrt(self.weights) * self.base_radius

        if np.any(close):
            center = (np.sum((self.means[close].T * self.weights[close]).T, axis = 0) + mean) / (np.sum(self.weights[close]) + 1)
            weight = min(np.sum(self.weights[close]) + 1, self.max_weight)
            self.means = np.concatenate([
                np.zeros((self.means.shape[0] - (np.sum(~close) + 1), self.means.shape[1])),
                self.means[~close],
                [ center ],
            ])
            self.weights = np.concatenate([
                np.zeros((self.weights.shape[0] - (np.sum(~close) + 1),)),
                self.weights[~close],
                [ weight ],
            ])
            return False
        else:
            self.means = np.concatenate([
                self.means[1:],
                [ mean ],
            ])
            self.weights = np.concatenate([
                self.weights[1:],
                [ 1 ],
            ])
            return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type = str, required = True)
    parser.add_argument('-o', '--output', type = str, required = True)
    parser.add_argument('-s', '--seed', type = int, default = 0)
    parser.add_argument('-f', '--force', action = 'store_true')
    args = parser.parse_args()

    random.seed(args.seed)

    print(f'gpu enabled: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    encoder = vae.Encoder(embedding_size = 16).to(device)
    encoder.load_state_dict(torch.load('mfcc-untested-1/encoder-F16-A0.9-E256-L171.pt'))
    encoder.eval()

    def prep_sample(x):
        s = mfcc.mfcc_spectrogram_for_learning(x, dataloader.UNIFORM_SAMPLE_RATE)
        with torch.no_grad():
            encoder_input = torch.tensor(s.reshape(1, *s.shape), dtype=torch.float32).to(device)
            mean, logstd = encoder.forward(encoder_input)
            assert mean.shape == (1, encoder.embedding_size)
            return np.array(mean.cpu()).reshape(-1), np.array(logstd.exp().cpu()).reshape(-1)

    filter = globals()[FILTER_TYPE](MAX_CLUSTERS, MAX_WEIGHT, encoder.embedding_size, FILTER_THRESH)

    if os.path.exists(args.output):
        if args.force:
            shutil.rmtree(args.output)
        else:
            print(f'output path \'{args.output}\' already exists', file = sys.stderr)
            sys.exit(1)

    keep_dir = f'{args.output}/keep'
    discard_dir = f'{args.output}/discard'
    os.makedirs(keep_dir)
    os.makedirs(discard_dir)

    dataset = DeploymentDataloader(args.input)
    kept = 0
    for i, (clip, chunks) in enumerate(dataset):
        votes = [1 if filter.insert(*prep_sample(chunk)) else 0 for chunk in chunks]
        vote_ratio = sum(votes) / len(votes)

        keep = vote_ratio >= VOTE_THRESH
        kept += 1 if keep else 0
        path = f'{keep_dir if keep else discard_dir}/{i:08}-{vote_ratio:.4f}.wav'
        print(f'{i:8} (ret {kept / (i + 1):.4f}) > {path}')
        soundfile.write(path, clip, dataloader.UNIFORM_SAMPLE_RATE)
