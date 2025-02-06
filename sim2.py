import argparse
import librosa
import dataloader
import random
import math
import os
import torch
import soundfile
import mfcc_vae_1 as vae
import mfcc
import numpy as np

from typing import Dict, Any

class ClusterFilterAug3:
    def __init__(self, max_clusters, max_weight, embedding_size, thresh):
        self.means = np.zeros((max_clusters, embedding_size))
        self.weights = np.zeros((max_clusters,))
        self.max_weight = max_weight
        self.max_clusters = max_clusters
        self.base_radius = thresh

    def insert(self, mean: np.ndarray, std: np.ndarray) -> bool:
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

def load_sounds(path: str, *, min_length: float = 0, max_length: float = math.inf, mult_length: float = None) -> Dict[str, np.ndarray]:
    res = { cls: [] for cls in sorted(os.listdir(path)) }
    for cls, entries in res.items():
        for file in sorted(os.listdir(f'{path}/{cls}')):
            clip, sr = librosa.load(f'{path}/{cls}/{file}', sr = dataloader.UNIFORM_SAMPLE_RATE)
            assert sr == dataloader.UNIFORM_SAMPLE_RATE, sr
            assert len(clip.shape) == 1, clip.shape
            if mult_length is not None:
                assert len(clip) % (mult_length * sr) == 0, f'{path}/{cls}/{file} not a multiple of {mult_length}s'

            if len(clip) < min_length * sr:
                print(f'  omitting {path}/{cls}/{file} -- too short ({len(clip) / sr:0.2f}s < {min_length:0.2f}s)')
                continue
            if len(clip) > max_length * sr:
                print(f'  omitting {path}/{cls}/{file} -- too long ({len(clip) / sr:0.2f}s > {max_length:0.2f}s)')
                continue

            entries.append(clip)
    return { k: v for k,v in res.items() if len(v) > 0 }

def random_contract(x: np.ndarray, s: int) -> np.ndarray:
    if x.shape[0] <= s: return x
    t = random.randrange(x.shape[0] - s)
    return x[t:t+s]
def random_extend(x: np.ndarray, s: int) -> np.ndarray:
    if x.shape[0] >= s: return x
    t = random.randrange(s - x.shape[0])
    res = np.zeros((s,))
    res[t:t+x.shape[0]] = x
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type = str, nargs = '+', required = True)
    parser.add_argument('--backgrounds', type = str, nargs = '+', required = True)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--clip-duration', type = float, default = 30)
    parser.add_argument('--clips', type = int, default = 2 * 60 * 24)
    parser.add_argument('--bg-change-prob', type = float, default = 0.1)
    parser.add_argument('--event-prob', type = float, default = 0.25)
    parser.add_argument('--event-freqs', type = str, nargs = '*', default = [])
    parser.add_argument('--max-clusters', type = int, default = 64)
    parser.add_argument('--max-weight', type = float, default = 4.0)
    parser.add_argument('--filter-thresh', type = float, default = 0.5)
    parser.add_argument('--vote-thresh', type = float, default = 0.0)
    parser.add_argument('--audio-out', type = str)
    args = parser.parse_args()

    if args.seed is not None: random.seed(args.seed)

    print(f'gpu enabled: {torch.cuda.is_available()}\n')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    encoder = vae.Encoder(embedding_size = 16).to(device)
    encoder.load_state_dict(torch.load('mfcc-untested-1/encoder-F16-A0.9-E256-L171.pt', weights_only = True))
    encoder.eval()

    print('loading sounds...')
    backgrounds = dict(sum((list(load_sounds(path, min_length = dataloader.SAMPLE_DURATION_SECS, mult_length = dataloader.SAMPLE_DURATION_SECS).items()) for path in sorted(args.backgrounds)), start = []))
    events = dict(sum((list(load_sounds(path, max_length = args.clip_duration).items()) for path in sorted(args.events)), start = []))
    print('loading complete\n')

    print(f'backgrounds: {({ k: len(v) for k,v in backgrounds.items() })}')
    print(f'events: {({ k: len(v) for k,v in events.items() })}\n')

    event_freqs = { x[:x.index(':')]: float(x[x.index(':')+1:]) for x in args.event_freqs }
    if '*' in event_freqs:
        event_freqs = { **event_freqs, **{ x: event_freqs['*'] for x in events.keys() if x not in event_freqs } }
        del event_freqs['*']
    event_freqs = { k: v for k, v in event_freqs.items() if v > 0 }
    if len(event_freqs) == 0: event_freqs = { x: 1 for x in events.keys() }
    for x in event_freqs.keys():
        if x not in events:
            raise RuntimeError(f'unknown event type: "{x}"')
    def pick_event():
        t = sum(event_freqs.values())
        r = random.random()
        p = 0
        e = None
        for event, weight in event_freqs.items():
            e = event
            p += weight / t
            if r < p: break
        return e

    f = ClusterFilterAug3(args.max_clusters, args.max_weight, encoder.embedding_size, args.filter_thresh)
    def vote_retain(x: np.ndarray) -> bool:
        segs = len(x) / (dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS)
        assert segs == float(int(segs)), segs
        votes = []
        for seg in np.split(x, segs):
            with torch.no_grad():
                mean, std = encoder.forward(torch.tensor(mfcc.mfcc_spectrogram_for_learning(seg, dataloader.UNIFORM_SAMPLE_RATE)[np.newaxis,:], dtype = torch.float).to(device))
                votes.append(f.insert(mean.cpu().numpy().squeeze(), std.cpu().numpy().squeeze()))
        return np.mean(votes) > args.vote_thresh

    clips = []
    input_events = { x: 0 for x in [None] + list(event_freqs.keys()) }
    output_events = input_events.copy()
    background_class = None
    clip_len = dataloader.UNIFORM_SAMPLE_RATE * args.clip_duration
    for i in range(args.clips):
        if background_class is None or random.random() < args.bg_change_prob:
            background_class = random.choice(sorted(backgrounds.keys()))
        background = random.choice(backgrounds[background_class])
        clip = np.tile(background, math.ceil(clip_len / len(background)))[:clip_len] # avoid random_contract to prevent transitions in inference chunks
        assert clip.shape == (clip_len,), clip.shape

        event_class = None
        if random.random() < args.event_prob:
            event_class = pick_event()
            event = random_contract(random_extend(random.choice(events[event_class]), clip_len), clip_len)
            clip += event

        clips.append(clip)
        input_events[event_class] += 1
        if vote_retain(clip): output_events[event_class] += 1

    for event in sorted(input_events.keys(), key = lambda x: -input_events[x]):
        print(f'{str(event):>40}: {input_events[event]:>5} -> {output_events[event]:>5} ({100 * output_events[event] / input_events[event] if input_events[event] != 0 else 0:>5.1f}%)')

    if args.audio_out is not None:
        soundfile.write(args.audio_out, np.concatenate(clips), samplerate = dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
