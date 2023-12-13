import numpy as np
import librosa
import random
import os

DATASET_EVENT_DURATION_SECS = 2
FILTERING_SLICE_DURATION_SECS = 1
UNIFORM_SAMPLE_RATE = 8000
MAX_TESTS_PER_CLASS = 5 # how many (random) examples to take (at most) from each class for testing the filters

def load_dataset_file(path: str):
    p = path.rfind('.')
    if p >= 0: path = path[:p]

    orig_samples, orig_sr = librosa.load(f'{path}.wav', sr = None)
    new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = UNIFORM_SAMPLE_RATE)

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

class FFTFilter:
    def __init__(self, init_sample, fold_ratio):
        self.background_freq = np.fft.fft(init_sample)
        self.energy_thresh = 0
        self.fold_ratio = fold_ratio
    def step(self, sample):
        freq = np.fft.fft(sample)
        energy = np.sum((freq - self.background_freq)**2)
        keep = energy > self.energy_thresh

        self.background_freq = (1 - self.fold_ratio) * self.background_freq + self.fold_ratio * freq
        self.energy_thresh = (1 - self.fold_ratio) * self.energy_thresh + self.fold_ratio * energy

        return keep

if __name__ == '__main__':
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
        filter = FFTFilter(segments[0][0], 0.1)
        correct = 0
        for segment, expected in segments:
            got = filter.step(segment)
            if got == expected: correct += 1
        total_correct += correct
        total_count += len(segments)
        print(f'test {test}: {correct / len(segments):.4f}')
    print(f'\navg: {total_correct / total_count:.4f}')