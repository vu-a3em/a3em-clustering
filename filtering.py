import numpy as np
import librosa
import random
import sys
import os
import torch

DATASET_EVENT_DURATION_SECS = 2
FILTERING_SLICE_DURATION_SECS = 1
UNIFORM_SAMPLE_RATE = 20000

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
    dt = round(FILTERING_SLICE_DURATION_SECS * UNIFORM_SAMPLE_RATE)
    edt = round(DATASET_EVENT_DURATION_SECS * UNIFORM_SAMPLE_RATE)
    while True:
        if t + dt > new_samples.shape[0]: break
        has_event = any(t + dt > raw_event[1] and t < raw_event[1] + edt for raw_event in raw_events)
        res.append((new_samples[t:t+dt], has_event))
        t += dt
    return res

class ConstFilter:
    def __init__(self, init_sample, v):
        self.v = v
    def forward(self, sample):
        return self.v

class RNGFilter:
    def __init__(self, init_sample, p):
        self.p = p
    def forward(self, sample):
        return random.random() > self.p

class FFTFilter:
    def __init__(self, init_sample, fold_ratio, energy_ratio):
        self.background_freq = np.abs(np.fft.rfft(init_sample))
        self.background_power = np.zeros(self.background_freq.shape)
        self.energy_thresh = 0
        self.fold_ratio = fold_ratio
        self.energy_ratio = energy_ratio
    def forward(self, sample):
        freq = np.abs(np.fft.rfft(sample))
        power = (freq - self.background_freq)**2
        energy = np.sum(power * (power > self.background_power * self.energy_ratio))
        keep = energy > self.energy_thresh * self.energy_ratio

        self.background_freq = (1 - self.fold_ratio) * self.background_freq + self.fold_ratio * freq
        self.background_power = (1 - self.fold_ratio) * self.background_power + self.fold_ratio * power
        self.energy_thresh = (1 - self.fold_ratio) * self.energy_thresh + self.fold_ratio * energy

        return keep

class LearningFFTFilter(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size, 16)
        self.classify = torch.nn.Linear(16, output_size)
    def forward(self, sample):
        freq = torch.abs(torch.fft.rfft(sample))
        x = torch.sigmoid(self.layer_1(freq))
        return torch.sigmoid(self.classify(x))

if __name__ == '__main__':
    print(f'gpu enabled: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dataset_root = 'dataset-partial'
    train = []
    test = []
    for label in os.listdir(dataset_root):
        paths = []
        for dirpath, dirnames, filenames in os.walk(f'{dataset_root}/{label}'):
            paths.extend(f'{dirpath}/{x}' for x in filenames if x.endswith('.wav'))
        random.shuffle(paths)
        train.extend(paths[:len(paths)//2])
        test.extend(paths[len(paths)//2:])

    filter = LearningFFTFilter(10001, 1).to(device)
    opt = torch.optim.Adam(filter.parameters(), lr = 2e-5)
    loss_fn = torch.nn.MSELoss()
    batch_size = 8
    epochs = 64

    def do_training_epoch(epoch):
        filter.train(True)
        losses = []
        batch_in = []
        batch_expect = []
        for i in range(len(train)):
            segments = load_dataset_file(train[i])
            if len(segments) == 0: continue
            for segment, expected in segments:
                batch_in.append(segment)
                batch_expect.append([1 if expected else 0])

                if len(batch_in) == batch_size:
                    assert len(batch_in) == len(batch_expect)

                    opt.zero_grad()
                    batch_out = filter.forward(torch.Tensor(batch_in).to(device))
                    loss = loss_fn(batch_out, torch.Tensor(batch_expect).to(device))
                    loss.backward()
                    losses.append(loss.item())
                    opt.step()

                    batch_in.clear()
                    batch_expect.clear()
            print(f'\rtraining epoch {epoch+1}/{epochs}... {i}/{len(train)} ({100*i/len(train):.2f}%)', end = '')
        print(f'\repoch {epoch+1}/{epochs}: avg loss {np.mean(losses)}                                         ')

    def do_testing():
        correct = 0
        total = 0
        filter.train(False)
        for i in range(len(test)):
            segments = load_dataset_file(test[i])
            if len(segments) == 0: continue
            for segment, expected in segments:
                got = filter.forward(torch.Tensor([segment]).to(device))[0][0] > 0.5
                if got == expected: correct += 1
                total += 1
            print(f'\rrunning tests... {i}/{len(test)} ({100*i/len(test):.2f}%)', end = '')
        print(f'\rtest accuracy: {correct / total}                                       ')

    for i in range(epochs):
        do_training_epoch(i)
        do_testing()

    # make_filter = lambda s: ConstFilter(s, True) # ~90%
    # make_filter = lambda s: RNGFilter(s, 0.1) # ~81%
    # make_filter = lambda s: FFTFilter(s, 0.75, 0.8)

    # total_correct = 0
    # total_count = 0
    # for test in tests:
    #     segments = load_dataset_file(test)
    #     if len(segments) == 0: continue
    #     filter = make_filter(segments[0][0])
    #     correct = 0
    #     for segment, expected in segments:
    #         got = filter.forward(segment)
    #         if got == expected: correct += 1
    #     total_correct += correct
    #     total_count += len(segments)
    #     # print(f'test {test}: {correct / len(segments):.4f}')
    # print(f'avg: {total_correct / total_count:.4f}')