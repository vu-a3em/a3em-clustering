# https://zenodo.org/record/5412896/files/Development_Set.zip?download=1

import argparse
import dataloader
import librosa
import soundfile
import os

class DcaseDataloader:
    def __init__(self, input: str, *, min_event_duration: float):
        self.input = input
        self.min_event_duration = min_event_duration

    def __iter__(self):
        for root, dirs, files in os.walk(self.input):
            for file in files:
                if not file.endswith('.wav'): continue
                wav, sr = librosa.load(f'{root}/{file}', sr = dataloader.UNIFORM_SAMPLE_RATE)
                assert sr == dataloader.UNIFORM_SAMPLE_RATE

                with open(f'{root}/{file[:-4]}.csv', 'r') as f:
                    i = iter(f)
                    headers = next(i).strip().split(',')
                    for row in i:
                        data = row.strip().split(',')
                        assert len(headers) == len(data), (len(headers), len(data))

                        labels = [headers[i] for i in range(3, len(data)) if data[i] == 'POS']
                        if len(labels) != 1: continue
                        if float(data[2]) - float(data[1]) < self.min_event_duration: continue
                        clip = wav[round(float(data[1]) * sr) : round(float(data[2]) * sr)]

                        yield clip, labels[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str)
    parser.add_argument('output', type = str)
    parser.add_argument('--min-event-duration', type = float, default = 0.5)
    args = parser.parse_args()

    os.mkdir(args.output)
    counts = {}
    for clip, label in DcaseDataloader(args.input, min_event_duration = args.min_event_duration):
        if label not in counts:
            counts[label] = 0
            os.mkdir(f'{args.output}/{label}')
        soundfile.write(f'{args.output}/{label}/{counts[label]}.wav', clip, dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
        counts[label] += 1
