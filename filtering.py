import numpy as np
import librosa
import random
import sys
import os

DATASET_EVENT_DURATION_SECS = 2
FILTERING_SLICE_DURATION_SECS = 1
UNIFORM_SAMPLE_RATE = 20000
MAX_TESTS_PER_CLASS = 20 # how many (random) examples to take (at most) from each class for testing the filters

def load_dataset_file(path: str):
    p = path.rfind('.')
    if p >= 0: path = path[:p]

    try:
        orig_samples, orig_sr = librosa.load(f'{path}.wav', sr = None)
        new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = UNIFORM_SAMPLE_RATE)
    except Exception as e:
        print(f'failed to read {path}.wav', file = sys.stderr)
        raise e

    raw_events = []
    with open(f'{path}.meta', 'r') as f:
        for line in f:
            vals = [x.strip() for x in line.strip().split(',')]
            if len(vals) == 2:
                vals = ['TRAIN', *vals]
            assert len(vals) == 3
            vals[1] = round(int(vals[1]) * (UNIFORM_SAMPLE_RATE / orig_sr))
            raw_events.append(vals)
    raw_events.sort(key = lambda x: x[1])

    res = []
    t = 0
    dt = round(FILTERING_SLICE_DURATION_SECS * UNIFORM_SAMPLE_RATE)
    edt = round(DATASET_EVENT_DURATION_SECS * UNIFORM_SAMPLE_RATE)
    while True:
        if t + dt > new_samples.shape[0]: break
        has_event = any(t + dt > raw_event[1] and t < raw_event[1] + edt for raw_event in raw_events)
        res.append((new_samples[t:t+dt], has_event))
        t += dt
    return res

class ConstFilter:
    def __init__(self, init_sample, v):
        self.v = v
    def step(self, sample):
        return self.v

class RNGFilter:
    def __init__(self, init_sample, p):
        self.p = p
    def step(self, sample):
        return random.random() > self.p

class FFTFilter:
    def __init__(self, init_sample, fold_ratio, energy_ratio):
        self.background_freq = np.abs(np.fft.rfft(init_sample))
        self.background_power = np.zeros(self.background_freq.shape)
        self.energy_thresh = 0
        self.fold_ratio = fold_ratio
        self.energy_ratio = energy_ratio
    def step(self, sample):
        freq = np.abs(np.fft.rfft(sample))
        power = (freq - self.background_freq)**2
        energy = np.sum(power * (power > self.background_power * self.energy_ratio))
        keep = energy > self.energy_thresh * self.energy_ratio

        self.background_freq = (1 - self.fold_ratio) * self.background_freq + self.fold_ratio * freq
        self.background_power = (1 - self.fold_ratio) * self.background_power + self.fold_ratio * power
        self.energy_thresh = (1 - self.fold_ratio) * self.energy_thresh + self.fold_ratio * energy

        return keep

if __name__ == '__main__':
    # make_filter = lambda s: ConstFilter(s, True) # ~90%
    # make_filter = lambda s: RNGFilter(s, 0.1) # ~81%

    for a in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        vs = []
        for _ in range(8):
            make_filter = lambda s: FFTFilter(s, a, 0.5)

            dataset_root = 'dataset-partial'
            tests = []
            for label in os.listdir(dataset_root):
                paths = []
                for dirpath, dirnames, filenames in os.walk(f'{dataset_root}/{label}'):
                    paths.extend(f'{dirpath}/{x}' for x in filenames if x.endswith('.wav'))
                if len(paths) > MAX_TESTS_PER_CLASS:
                    paths = random.sample(paths, MAX_TESTS_PER_CLASS)
                assert len(paths) <= MAX_TESTS_PER_CLASS
                tests.extend(paths)

            total_correct = 0
            total_count = 0
            for test in tests:
                segments = load_dataset_file(test)
                if len(segments) == 0: continue
                filter = make_filter(segments[0][0])
                correct = 0
                for segment, expected in segments:
                    got = filter.step(segment)
                    if got == expected: correct += 1
                total_correct += correct
                total_count += len(segments)
                # print(f'test {test}: {correct / len(segments):.4f}')
            print(f'avg: {total_correct / total_count:.4f} ({a})')
            vs.append(total_correct / total_count)
        print(f'avg avg: {np.mean(vs)}')