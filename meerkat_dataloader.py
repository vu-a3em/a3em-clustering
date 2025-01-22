import argparse
import dataloader
import librosa
import h5py
import numpy as np
import soundfile
import sys
import os

class MeerkatDataloader:
    def __init__(self, path: str, *, delete_unlabeled: bool = False):
        self.path = path
        self.delete_unlabeled = delete_unlabeled

        self.entries = [x[:-3] for x in os.listdir(f'{self.path}/lbl/08000Hz') if x.endswith('.h5')]
        self.entries.sort()

    def __iter__(self):
        for entry in self.entries:
            with h5py.File(f'{self.path}/lbl/08000Hz/{entry}.h5', 'r') as f:
                start_frames = f['start_frame_lbl'][:]
                if len(start_frames) == 0:
                    if self.delete_unlabeled:
                        os.remove(f'{self.path}/lbl/08000Hz/{entry}.h5')
                        os.remove(f'{self.path}/wav/08000Hz/{entry}.wav')
                    continue
                stop_frames = f['end_frame_lbl'][:]
                labels = f['lbl'][:]
                assert len(start_frames) == len(stop_frames) == len(labels)
            clip, sr = librosa.load(f'{self.path}/wav/08000Hz/{entry}.wav', sr = None)
            assert sr == dataloader.UNIFORM_SAMPLE_RATE
            s = dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS
            assert clip.shape[0] % s == 0

            clips = np.split(clip, clip.shape[0] / s)
            assert all([c.shape[0] == s for c in clips])

            clip_labels = [set() for _ in clips]
            for i, clip_label in enumerate(clip_labels):
                for j in range(len(labels)):
                    if start_frames[j] / s >= i and stop_frames[j] / s <= (i + 1):
                        clip_label.add(labels[j])

            for i in range(len(clips)):
                if len(clip_labels[i]) == 1:
                    yield clips[i], next(iter(clip_labels[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str)
    parser.add_argument('output', type = str)
    args = parser.parse_args()

    counts = {}
    os.mkdir(args.output)
    for clip, label in MeerkatDataloader(args.input):
        if label not in counts:
            counts[label] = 0
            os.mkdir(f'{args.output}/{label}')
        soundfile.write(f'{args.output}/{label}/{counts[label]}.wav', clip, dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
        counts[label] += 1
