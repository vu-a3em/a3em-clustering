import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np

import argparse
import json

from util import load_sound

parser = argparse.ArgumentParser()
parser.add_argument('source')
parser.add_argument('labels')
args = parser.parse_args()

with open(args.labels, 'r') as f:
    labels = json.load(f)

source = load_sound(args.source)
sample_rate, samples = wav.read(args.source)
channels = [samples] if len(samples.shape) == 1 else [samples[:,i] for i in range(samples.shape[1])]

for i, channel in enumerate(channels):
    plt.plot(np.arange(len(channel)) / sample_rate, channel, label = f'channel {i + 1}')
plt.xlabel('time (s)')
plt.legend(loc = 'upper left')
plt.show()
