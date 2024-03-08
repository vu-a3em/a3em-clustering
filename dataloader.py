import librosa
import random
import sys
import os

import scipy.signal as sig
import numpy as np

DATASET_EVENT_DURATION_SECS = 2
SAMPLE_DURATION_SECS = 1
UNIFORM_SAMPLE_RATE = 8000 # 20000

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
            if vals[2].lower() == 'ignore': continue
            vals[1] = round(int(vals[1]) * (UNIFORM_SAMPLE_RATE / orig_sr))
            raw_events.append(vals)
    raw_events.sort(key = lambda x: x[1])

    res = []
    t = 0
    dt = round(SAMPLE_DURATION_SECS * UNIFORM_SAMPLE_RATE)
    edt = round(DATASET_EVENT_DURATION_SECS * UNIFORM_SAMPLE_RATE)
    while True:
        if t + dt > new_samples.shape[0]: break
        label = 'BackgroundSounds'
        for raw_event in raw_events:
            if t + dt > raw_event[1] and t < raw_event[1] + edt:
                label = raw_event[2]
                break
        res.append((label, new_samples[t:t+dt]))
        t += dt
    return res

def get_dataset(max_files_per_class, max_events_per_class):
    dataset = {}
    root = 'dataset-partial'
    for cls in os.listdir(root):
        files = []
        for dirpath, dirnames, filenames in os.walk(f'{root}/{cls}'):
            for filename in filenames:
                if filename.endswith('.wav'):
                    files.append(f'{dirpath}/{filename}')
        random.shuffle(files)
        for i, file in enumerate(files):
            if max_files_per_class is not None and i >= max_files_per_class: break
            sub_dataset = {}
            for label, data in load_dataset_file(file):
                if label not in sub_dataset:
                    sub_dataset[label] = []
                sub_dataset[label].append(data)
            for k, v in sub_dataset.items():
                random.shuffle(v)
                if k not in dataset:
                    dataset[k] = []
                dataset[k].extend(v)
    for k in dataset.keys():
        random.shuffle(dataset[k])
        if max_events_per_class is not None and len(dataset[k]) > max_events_per_class:
            dataset[k] = dataset[k][:max_events_per_class]
    return dataset

# def get_dataset(max_files_per_class, max_events_per_class):
#     dataset = {}
#     for dirpath, dirnames, filenames in os.walk('dataset-partial'):
#         random.shuffle(filenames)
#         for i, filename in enumerate(filenames):
#             if i >= max_files_per_class: break
#             if not filename.endswith('.wav'): continue
#             sub_dataset = {}
#             for label, data in load_dataset_file(f'{dirpath}/{filename}'):
#                 if label not in sub_dataset:
#                     sub_dataset[label] = []
#                 sub_dataset[label].append(data)
#             for k, v in sub_dataset.items():
#                 random.shuffle(v)
#                 if k not in dataset:
#                     dataset[k] = []
#                 dataset[k].extend(v)
#     for k in dataset.keys():
#         random.shuffle(dataset[k])
#         if len(dataset[k]) > max_events_per_class:
#             dataset[k] = dataset[k][:max_events_per_class]
#     return dataset

def get_spectrogram(sample):
    assert 1 <= len(sample.shape) <= 2
    if len(sample.shape) == 2:
        sample = np.average(sample, axis = 0)
    return sig.spectrogram(sample, UNIFORM_SAMPLE_RATE)
