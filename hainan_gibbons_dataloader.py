# https://zenodo.org/record/3991714/files/Train.zip?download=1
# https://zenodo.org/record/3991714/files/Train_Labels.zip?download=1

import os
import argparse
import librosa
import soundfile
import dataloader
import os

class HainanGibbonsDataloader:
    def __init__(self, *, audio_path: str, labels_path: str, max_length: float):
        self.audio_path = audio_path
        self.labels_path = labels_path
        self.max_length = max_length
    def __iter__(self):
        for meta_file in os.listdir(self.labels_path):
            assert any(meta_file.startswith(x) for x in ['g_', 'n_']) and meta_file.endswith('.data')
            master_clip, sr = librosa.load(f'{self.audio_path}/{meta_file[2:-5]}.wav', sr = dataloader.UNIFORM_SAMPLE_RATE)
            assert sr == dataloader.UNIFORM_SAMPLE_RATE, sr
            with open(f'{self.labels_path}/{meta_file}', 'r') as f:
                header = next(f).strip().split(',')
                for meta in f:
                    meta = meta.strip().split(',')
                    assert len(meta) == len(header)
                    meta = { header[i]: meta[i] for i in range(len(meta)) }
                    if meta['Type'] == '': continue
                    t1, t2 = float(meta['Start']), float(meta['End'])
                    if t2 <= t1: continue
                    if t2 - t1 > self.max_length: continue
                    yield master_clip[round(t1 * sr) : round(t2 * sr)], meta['Type']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_input', type = str)
    parser.add_argument('labels_input', type = str)
    parser.add_argument('output', type = str)
    parser.add_argument('--max-length', type = float, default = 30.0)
    args = parser.parse_args()

    os.mkdir(args.output)
    counts = {}
    for clip, label in HainanGibbonsDataloader(audio_path = args.audio_input, labels_path = args.labels_input, max_length = args.max_length):
        if label not in counts:
            os.mkdir(f'{args.output}/{label}')
            counts[label] = 0
        soundfile.write(f'{args.output}/{label}/{counts[label]}', clip, samplerate = dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
        counts[label] += 1
