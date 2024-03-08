import argparse
import librosa
import pyaudio
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', required = True)
parser.add_argument('-o', '--output')
parser.add_argument('-d', '--duration', type = float, required = True)
parser.add_argument('-w', '--width', type = int, default = 4)
parser.add_argument('-r', '--resume', action = 'store_true')
args = parser.parse_args()

if args.output is None:
    args.output = f'{args.input[:args.input.rfind(".")]}.csv'

def play_audio(content, sr, width):
    p = pyaudio.PyAudio()
    stream = p.open(format = p.get_format_from_width(width), channels = 1, rate = sr, output = True, frames_per_buffer = 1024)
    stream.write(content.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

content, sr = librosa.load(args.input, sr = None)
assert len(content.shape) == 1
print(f'loaded "{args.input}" -- duration: {content.shape[0] / sr:.2f} seconds')
dt = round(args.duration * sr)
assert dt > 0
labels = set()

start_time = 0
if args.resume:
    with open(args.output, 'r') as f:
        for line in f:
            x = line.strip()
            if x == '': continue

            x = line.split(',')
            start_time = round(float(x[1]) * sr)
            labels.add(x[2].strip())
else:
    if os.path.exists(args.output):
        print(f'output file \'{args.output}\' already exists', file = sys.stderr)
        sys.exit(0)

with open(args.output, 'w' if not args.resume else 'a') as f:
    t = start_time
    i = 0
    while t + dt < content.shape[0]:
        print(f'segment {i} (t = {t / sr:.2f}..{(t + dt) / sr:.2f}) ({100 * t / content.shape[0]:.4f}%)')
        seg = content[t:t + dt]

        while True:
            play_audio(seg, sr, args.width)
            label = input('label: ')
            if label is None or label == '': continue
            if label in labels: break

            while True:
                c = input(f'unknown label "{label}" - knowns labels are {labels}\nadd "{label}" as a new label type? (y/n)').lower()
                if c in ['y', 'yes', 'n', 'no']: break
            if c in ['y', 'yes']:
                labels.add(label)
                break

        f.write(f'{t / sr},{(t + dt) / sr},{label}\n')
        f.flush()
        t += dt
        i += 1
