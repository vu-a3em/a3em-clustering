import random
import csv
import re
import dataloader
import librosa
import argparse
import os
import sys
import soundfile

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

def crash(msg: str):
    print(msg, file = sys.stderr)
    sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str)
    parser.add_argument('output', type = str)
    args = parser.parse_args()

    all_classes = { x: x for x in [
        'Auto', 'Airc', 'Mach',
        'Mam', 'Bug', 'Wind', 'Rain', 'Water', 'Truck', 'Car',
        'Prop', 'Helo', 'Jet', 'Corv', 'DGS', 'Grous', 'Crane',
        'Loon', 'Owl', 'Hum', 'Rapt', 'Woop', 'ShorB', 'Woof',
        'Bear', 'Mous', 'Deer', 'Weas', 'Meow', 'Hare', 'Shrew', 'Mosq', 'Fly',
    ]}
    all_classes['Flare'] = 'Mach'

    if os.path.exists(args.output):
        crash(f'output path "{args.output}" already exists')

    dataset = EdansaDataloader(args.input)
    counts = { x: 0 for x in all_classes.values() }
    discarded_combos = set()
    empty_combos = set()
    total = 0
    for i, (clip, segments, meta) in enumerate(dataset):
        total += 1
        classes = sorted(set([y for x, y in all_classes.items() if meta[x]]))
        if len(classes) != 1:
            discarded_combos.add(tuple(classes))
            print(f'skipping entry {i}... classes={classes}')
            continue
        cls = classes[0]
        if counts[cls] == 0:
            os.makedirs(f'{args.output}/{cls}')
        soundfile.write(f'{args.output}/{cls}/{counts[cls]}.wav', clip, dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
        counts[cls] += 1
    print(f'\ndone: kept {sum(counts.values())}/{total} ({sum(counts.values()) / total:0.4f})')
    print(f'discarded_combos: {sorted(discarded_combos)}')
    print(f'counts: {counts}')
