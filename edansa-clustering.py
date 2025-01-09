import numpy as np
import argparse
import torch
import random
import os
import librosa

import dataloader
import mfcc
import mfcc_vae_1 as mfcc_vae

def mode(arr):
    counts = {}
    for v in arr:
        if v not in counts:
            counts[v] = 0
        counts[v] += 1
    m = max(counts.values())
    return random.choice([v for v,c in counts.items() if c == m])
def train_cluster_classifier(x, y, *, n_clusters):
    assert x.shape[0] == y.shape[0]
    clusters = x[np.random.choice(len(x), n_clusters)]
    prev_nearest_cluster = np.array([-1 for _ in range(len(x))])
    while True:
        nearest_cluster = np.array([np.argmin(np.sum((clusters - p)**2, axis = 1)) for p in x])
        assert nearest_cluster.shape == prev_nearest_cluster.shape
        if np.all(nearest_cluster == prev_nearest_cluster):
            break
        prev_nearest_cluster = nearest_cluster
        clusters = np.array([ np.mean([x[j] for j in range(len(x)) if nearest_cluster[j] == i], axis = 0) for i in range(len(clusters)) if np.any(nearest_cluster == i) ])
    cluster_labels = np.array([ mode([y[j] for j in range(len(x)) if prev_nearest_cluster[j] == i]) for i in range(len(clusters)) ])
    return clusters, cluster_labels

def cluster_classify(clusters, cluster_labels, value):
    return cluster_labels[np.argmin(np.sum((clusters - value)**2, axis = 1))]

def test_cluster_classifier(clusters, cluster_labels, x, y, num_labels):
    assert len(clusters) == len(cluster_labels) and len(x) == len(y)
    confusion = np.zeros((num_labels, num_labels))
    for i in range(len(x)):
        predict = cluster_classify(clusters, cluster_labels, x[i])
        confusion[y[i],predict] += 1
    return confusion

def print_confusion_matrix(v, label_seq):
    assert v.shape == (len(v), len(v))
    for i in range(len(v)):
        for j in range(len(v)):
            print(f'{round(v[i,j]):4d} ', end = '')
        print()
    print('labels:', label_seq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type = int, required = False)
    parser.add_argument('--input', type = str, required = True)
    parser.add_argument('--clusters', type = int, required = True)
    parser.add_argument('--max-examples', type = int, required = False)
    args = parser.parse_args()

    assert args.classes is None or args.classes >= 1, args.classes
    assert args.max_examples is None or args.max_examples >= 1, args.max_examples
    assert args.clusters >= 1, args.clusters

    print(f'gpu enabled: {torch.cuda.is_available()}')
    accel_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    raw_dataset = {}
    for cls in os.listdir(args.input):
        raw_dataset[cls] = []
        for entry in os.listdir(f'{args.input}/{cls}'):
            assert entry.endswith('.wav'), entry
            orig_samples, orig_sr = librosa.load(f'{args.input}/{cls}/{entry}', sr = None)
            new_samples = librosa.resample(orig_samples, orig_sr = orig_sr, target_sr = dataloader.UNIFORM_SAMPLE_RATE)
            assert len(new_samples.shape) == 1 and new_samples.shape[0] % (dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS) == 0, new_samples.shape
            for segment in np.split(new_samples, new_samples.shape[0] / (dataloader.UNIFORM_SAMPLE_RATE * dataloader.SAMPLE_DURATION_SECS)):
                raw_dataset[cls].append(segment)
        random.shuffle(raw_dataset[cls])
        if args.max_examples is not None:
            raw_dataset[cls] = raw_dataset[cls][:min(len(raw_dataset[cls]), args.max_examples)]
    if args.classes is not None:
        v = list(raw_dataset.items())
        random.shuffle(v)
        raw_dataset = dict(v[:min(len(v), args.classes)])
    classes_ordered = list(raw_dataset.keys())

    print(f'loaded {sum(len(v) for v in raw_dataset.values())} data points ({ { k: len(v) for k, v in raw_dataset.items() } })')

    for quantize in [False, True]:
        print(f'\n{"=" * 25}\n== quantization: {"true" if quantize else "false":>5} ==\n{"=" * 25}\n')

        if quantize:
            encoder = torch.jit.load('portable-model-i8.pt')
            device = 'cpu'
        else:
            encoder = mfcc_vae.Encoder(embedding_size = 16).to(accel_device)
            encoder.load_state_dict(torch.load('mfcc-untested-3/encoder-F16-A0.999-E256-L183.pt', weights_only = True))
            device = accel_device
        encoder.eval()

        def prep_sample(x):
            with torch.no_grad():
                encoder_input = torch.tensor(mfcc.mfcc_spectrogram_for_learning(x, dataloader.UNIFORM_SAMPLE_RATE)).to(device).float()
                mean, logstd = encoder.forward(encoder_input.reshape(1, *encoder_input.shape))
                assert mean.shape == (1, 16)
                return np.array(mean.cpu()).reshape(-1)

        LABEL_TO_INT = {}
        INT_TO_LABEL = {}
        def prep_dataset(data):
            X = []
            Y = []
            for i, (label, samples) in enumerate(data.items()):
                LABEL_TO_INT[label] = i
                INT_TO_LABEL[i] = label
                for sample in samples:
                    X.append(prep_sample(sample))
                    Y.append(i)
            return np.array(X), np.array(Y)
        X_train, Y_train = prep_dataset({ k: v[:len(v) // 2] for k, v in raw_dataset.items() })
        X_eval, Y_eval = prep_dataset({ k: v[len(v) // 2:] for k, v in raw_dataset.items() })

        clusters, cluster_labels = train_cluster_classifier(X_train, Y_train, n_clusters = args.clusters)
        print('trained cluster classifier:', clusters.shape, cluster_labels)
        print()

        confusion = test_cluster_classifier(clusters, cluster_labels, X_train, Y_train, len(raw_dataset))
        print(f'train accuracy: {np.einsum("ii", confusion) / len(X_train):.2f} ({round(np.einsum("ii", confusion))} / {len(X_train)})')
        print('confusion:')
        print_confusion_matrix(confusion, classes_ordered)
        print()

        confusion = test_cluster_classifier(clusters, cluster_labels, X_eval, Y_eval, len(raw_dataset))
        print(f'eval accuracy: {np.einsum("ii", confusion) / len(X_eval):.2f} ({round(np.einsum("ii", confusion))} / {len(X_eval)})')
        print('confusion:')
        print_confusion_matrix(confusion, classes_ordered)
        print()
