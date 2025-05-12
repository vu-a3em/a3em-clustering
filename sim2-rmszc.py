import argparse
import librosa
import dataloader
import random
import math
import os
import soundfile
import mfcc
import numpy as np
import filter

from typing import Dict, Any, Optional

QUIET = False
def qprint(*args, **kwargs):
    if not QUIET:
        print(*args, **kwargs)

def load_sounds(path: str, *, min_length: float = 0, max_length: float = math.inf, mult_length: float = None, max_silence_ratio: float = None) -> Dict[str, np.ndarray]:
    res = { cls: [] for cls in sorted(os.listdir(path)) }
    for cls, entries in res.items():
        for file in sorted(os.listdir(f'{path}/{cls}')):
            clip, sr = librosa.load(f'{path}/{cls}/{file}', sr = dataloader.UNIFORM_SAMPLE_RATE)
            assert sr == dataloader.UNIFORM_SAMPLE_RATE, sr
            assert len(clip.shape) == 1, clip.shape
            if mult_length is not None:
                assert len(clip) % (mult_length * sr) == 0, f'{path}/{cls}/{file} not a multiple of {mult_length}s'

            if len(clip) < min_length * sr:
                qprint(f'  omitting {path}/{cls}/{file} -- too short ({len(clip) / sr:0.2f}s < {min_length:0.2f}s)')
                continue
            if len(clip) > max_length * sr:
                qprint(f'  omitting {path}/{cls}/{file} -- too long ({len(clip) / sr:0.2f}s > {max_length:0.2f}s)')
                continue

            if max_silence_ratio is not None and np.mean(clip == 0) > max_silence_ratio:
                qprint(f'  omitting {path}/{cls}/{file} -- too much silence')
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

def create_fade(size, *, fade_duration, sr):
    t = min(round(fade_duration * sr), round(size / 4))
    return np.concatenate([
        np.linspace(0, 1, t),
        np.ones((size - 2 * t,)),
        np.linspace(1, 0, t),
    ])

class RMSZCEncoder:
    def __init__(self, *, sample_rate: int):
        self.sample_rate = sample_rate
    def forward(self, x: np.ndarray) -> np.ndarray:
        raw = np.array([np.mean(librosa.feature.rms(y = x)), np.mean(librosa.feature.zero_crossing_rate(x))])
        return (raw - np.array([0.12521662, 0.03039862])) / np.array([0.14202671, 0.05519523])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type = str, nargs = '+', required = True)
    parser.add_argument('--backgrounds', type = str, nargs = '+', required = True)
    parser.add_argument('--seed', type = int)
    parser.add_argument('--clip-duration', type = float, default = 30)
    parser.add_argument('--clips', type = int, default = 10)
    parser.add_argument('--bg-change-prob', type = float, default = 0.01666)
    parser.add_argument('--event-prob', type = float, default = 1.0)
    parser.add_argument('--event-freqs', type = str, nargs = '*', default = [])
    parser.add_argument('--max-clusters', type = int, default = 256)
    parser.add_argument('--max-weight', type = float, default = 1024.0)
    parser.add_argument('--filter-thresh', type = float, default = 0.2)
    parser.add_argument('--audio-out', type = str)
    parser.add_argument('--fade-duration', type = float, default = 1)
    parser.add_argument('--max-silence-ratio', type = float, default = 0.1)
    parser.add_argument('--background-scale', type = float, default = 1.0)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--quiet', action = 'store_true')
    args = parser.parse_args()

    assert args.iterations >= 1
    if args.seed is not None: random.seed(args.seed)
    globals()['QUIET'] = True

    qprint('loading sounds...')
    backgrounds = dict(sum((list(load_sounds(path, min_length = dataloader.SAMPLE_DURATION_SECS, mult_length = dataloader.SAMPLE_DURATION_SECS, max_silence_ratio = args.max_silence_ratio).items()) for path in sorted(args.backgrounds)), start = []))
    events = dict(sum((list(load_sounds(path, max_length = args.clip_duration, max_silence_ratio = args.max_silence_ratio).items()) for path in sorted(args.events)), start = []))
    qprint('loading complete\n')

    qprint(f'backgrounds: {({ k: len(v) for k,v in backgrounds.items() })}')
    qprint(f'events: {({ k: len(v) for k,v in events.items() })}\n')

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

    print(f'event freqs: {event_freqs}\n')

    for x in [x for x in events.keys() if x not in event_freqs]:
        del events[x]

    encoder = RMSZCEncoder(sample_rate = dataloader.UNIFORM_SAMPLE_RATE)

    clips = []
    input_events = { x: 0 for x in [None] + list(event_freqs.keys()) }
    output_events = input_events.copy()
    background_class = None
    clip_len = dataloader.UNIFORM_SAMPLE_RATE * args.clip_duration
    embedding_size = len(encoder.forward(np.zeros((clip_len,))))
    for i in range(args.iterations):
        f = filter.ClusterFilter(args.max_clusters, args.max_weight, embedding_size, args.filter_thresh)

        for _ in range(args.clips):
            if background_class is None or random.random() < args.bg_change_prob:
                background_class = random.choice(sorted(backgrounds.keys()))
            background = random.choice(backgrounds[background_class])
            clip = np.tile(background, math.ceil(clip_len / len(background)))[:clip_len] # avoid random_contract to prevent transitions in inference chunks
            assert clip.shape == (clip_len,), clip.shape

            clip *= args.background_scale

            event_class = None
            if random.random() < args.event_prob:
                event_class = pick_event()
                event = random.choice(events[event_class])
                event = event * create_fade(len(event), fade_duration = args.fade_duration, sr = dataloader.UNIFORM_SAMPLE_RATE)
                event = random_contract(random_extend(event, clip_len), clip_len)
                clip += event

            if args.audio_out is not None: clips.append(clip)
            input_events[event_class] += 1
            if f.insert(encoder.forward(clip)): output_events[event_class] += 1

        if args.audio_out is not None:
            p = args.audio_out if args.iterations == 1 else f'{args.audio_out[:args.audio_out.rfind(".")]}-{i}.{args.audio_out[args.audio_out.rfind(".")+1:]}'
            soundfile.write(args.audio_out, np.concatenate(clips), samplerate = dataloader.UNIFORM_SAMPLE_RATE, format = args.audio_out[args.audio_out.rfind('.')+1:].upper())

    for event in sorted(input_events.keys(), key = lambda x: -input_events[x]):
        print(f'{str(event):>40}: {input_events[event]:>5} -> {output_events[event]:>5} ({100 * output_events[event] / input_events[event] if input_events[event] != 0 else 0:>5.1f}%)')