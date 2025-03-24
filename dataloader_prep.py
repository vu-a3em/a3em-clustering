import dataloader
import argparse
import soundfile
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = str, required = True)
    parser.add_argument('--clip-duration', type = float, required = True)
    args = parser.parse_args()

    assert args.clip_duration > 0

    os.mkdir(args.output)
    for label, clips in dataloader.get_dataset(None, None, sample_duration = args.clip_duration).items():
        os.mkdir(f'{args.output}/{label}')
        for i, clip in enumerate(clips):
            soundfile.write(f'{args.output}/{label}/{i}.wav', clip, dataloader.UNIFORM_SAMPLE_RATE, format = 'WAV')
