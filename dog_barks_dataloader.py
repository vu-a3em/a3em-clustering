# https://storage.googleapis.com/ml-bioacoustics-datasets/dog_barks.zip

import argparse
import librosa
import dataloader
import soundfile
import os

class DogBarksDataloader:
    def __init__(self, src: str):
        self.src = src
    def __iter__(self):
        with open(f'{self.src}/annotations.csv', 'r') as f:
            header = next(f).strip().split(',')
            for row in f:
                meta = row.strip().split(',')
                assert len(meta) == len(header)
                meta = { header[i]: meta[i] for i in range(len(header)) }

                clip, sr = librosa.load(f'{self.src}/audio/{meta["filename"]}', sr = dataloader.UNIFORM_SAMPLE_RATE)
                assert sr == dataloader.UNIFORM_SAMPLE_RATE, sr

                yield clip, meta['breed']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str)
    parser.add_argument('output', type = str)
    args = parser.parse_args()

    os.mkdir(args.output)
    counts = {}
    for clip, label in DogBarksDataloader(args.input):
        if label not in counts:
            os.mkdir(f'{args.output}/{label}')
            counts[label] = 0
        soundfile.write(f'{args.output}/{label}/{counts[label]}.wav', clip, dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
        counts[label] += 1
