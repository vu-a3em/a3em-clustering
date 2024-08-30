import numpy as np
import argparse
import random
import librosa
import re
import csv
import torch

import dataloader
import mfcc
import mfcc_vae_1 as vae

# (ret 0.3077) (ret int 0.5684) (dis int 0.4696)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.25
# MAX_CLUSTERS = 256
# MAX_WEIGHT = 4
# VOTE_THRESH = 0.2

# (ret 0.1062) (ret int 0.6161) (dis int 0.4862)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.4
# MAX_CLUSTERS = 256
# MAX_WEIGHT = 4
# VOTE_THRESH = 0.2

# (ret 0.3037) (ret int 0.5660) (dis int 0.4712)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.25
# MAX_CLUSTERS = 512
# MAX_WEIGHT = 4
# VOTE_THRESH = 0.2

# (ret 0.1246) (ret int 0.7124) (dis int 0.4698)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.5
# MAX_CLUSTERS = 256
# MAX_WEIGHT = 8
# VOTE_THRESH = 0.01

# (ret 0.1252) (ret int 0.7113) (dis int 0.4697)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.5
# MAX_CLUSTERS = 128
# MAX_WEIGHT = 8
# VOTE_THRESH = 0.01

# (ret 0.1256) (ret int 0.7120) (dis int 0.4695)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.5
# MAX_CLUSTERS = 64
# MAX_WEIGHT = 8
# VOTE_THRESH = 0.01

# (ret 0.1391) (ret int 0.7092) (dis int 0.4662)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.6125
# MAX_CLUSTERS = 64
# MAX_WEIGHT = 5
# VOTE_THRESH = 0.01

# (ret 0.1167) (ret int 0.7239) (dis int 0.4704)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.6
# MAX_CLUSTERS = 64
# MAX_WEIGHT = 6
# VOTE_THRESH = 0.01

# (ret 0.1220) (ret int 0.7224) (dis int 0.4691)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.6125
# MAX_CLUSTERS = 64
# MAX_WEIGHT = 5.5
# VOTE_THRESH = 0.01

# (ret 0.1233) (ret int 0.7227) (dis int 0.4687)
# FILTER_TYPE = 'ClusterFilterAug3'
# FILTER_THRESH = 0.625
# MAX_CLUSTERS = 64
# MAX_WEIGHT = 5.2
# VOTE_THRESH = 0.01

FILTER_TYPE = 'ClusterFilterAug3'
FILTER_THRESH = 0.625
MAX_CLUSTERS = 64
MAX_WEIGHT = 5.2
VOTE_THRESH = 0.01

class EdansaDataloader:
    def __init__(self, path: str):
        def try_int(x):
            try:
                return int(x)
            except:
                return x

        sort_key = lambda x: [try_int(y) for y in re.split(r'(\d+)', x['Clip Path'])]

        with open(f'{path}/labels.csv') as f:
            r = csv.reader(f)
            header = next(r)
            data = []
            for row in r:
                assert len(row) == len(header)
                data.append({ header[i]: try_int(row[i]) for i in range(len(header)) })

        is_event = lambda x: x['Sil'] == 0
        events = [{ 'is_event': True, **x } for x in data if is_event(x)]
        not_events = [{ 'is_event': False, **x } for x in data if not is_event(x)]

        events.sort(key = sort_key) # sort to order by deployment then chronologically
        not_events.sort(key = sort_key) # sort to order by deployment then chronologically

        n = min(len(events), len(not_events))
        events = events[:n]
        not_events = not_events[:n]
        assert len(events) == len(not_events)

        self.__path = path
        self.__data = events + not_events

        self.__data.sort(key = sort_key) # sort to order by deployment then chronologically

    def __iter__(self):
        target_samples = dataloader.SAMPLE_DURATION_SECS * dataloader.UNIFORM_SAMPLE_RATE
        for entry in self.__data:
            file = f'{self.__path}/data/{entry["Clip Path"]}'
            try:
                orig_samples, orig_sr = librosa.load(file, sr = None)
                new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = dataloader.UNIFORM_SAMPLE_RATE)
            except Exception as e:
                print(f'failed to read file \'{file}\':\n{e}\nskipping...\n')
                continue

            clip_count = new_samples.shape[0] // target_samples
            front_split = random.randrange(new_samples.shape[0] % target_samples) if new_samples.shape[0] % target_samples != 0 else 0
            trimmed = new_samples[front_split : front_split + clip_count * target_samples]
            assert trimmed.shape[0] // target_samples == clip_count and trimmed.shape[0] % target_samples == 0

            yield trimmed, [trimmed[i * target_samples : (i + 1) * target_samples] for i in range(clip_count)], entry

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
                self.means[~close],
                [ center ],
                np.zeros((self.means.shape[0] - (np.sum(~close) + 1), self.means.shape[1])),
            ])
            self.weights = np.concatenate([
                self.weights[~close],
                [ weight ],
                np.zeros((self.weights.shape[0] - (np.sum(~close) + 1),)),
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
    parser.add_argument('-s', '--seed', type = int, default = 0)
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

    dataset = EdansaDataloader(args.input)
    confusion = [[0, 0], [0, 0]]
    for i, (clip, chunks, meta) in enumerate(dataset):
        votes = [1 if filter.insert(*prep_sample(chunk)) else 0 for chunk in chunks]
        vote_ratio = sum(votes) / len(votes)

        keep = vote_ratio >= VOTE_THRESH
        confusion[meta['is_event']][keep] += 1
        ret = (confusion[True][True] + confusion[False][True]) / (i + 1)
        ret_int = confusion[True][True] / max(1, confusion[True][True] + confusion[False][True])
        dis_int = confusion[True][False] / max(1, confusion[True][False] + confusion[False][False])
        print(f'{i:4} (ret {ret:.4f}) (ret int {ret_int:.4f}) (dis int {dis_int:.4f}) > {"keep   " if keep else "discard"} > { { k: meta[k] for k in ["Anth", "Bio", "Geo", "Sil"] } }')

    print('\nconfusion (actual x predicted):')
    for row in confusion:
        for col in row:
            print(f'{col:4} ', end = '')
        print()
