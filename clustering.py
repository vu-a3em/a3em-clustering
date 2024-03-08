import numpy as np
import argparse
import torch
import random

import dataloader
import vae_10 as vae

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
    parser.add_argument('-e', '--embedding-size', type = int, required = True)
    parser.add_argument('-w', '--weights', type = str, required = True)
    parser.add_argument('-c', '--classes', type = int, required = False)
    args = parser.parse_args()

    print(f'gpu enabled: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    encoder = vae.Encoder(embedding_size = args.embedding_size).to(device)
    encoder.load_state_dict(torch.load(args.weights))
    encoder.eval()

    LABEL_TO_INT = {}
    INT_TO_LABEL = {}
    def prep_sample(x):
        f, t, Sxx = dataloader.get_spectrogram(x)
        with torch.no_grad():
            encoder_input = torch.tensor(np.log10(np.maximum(Sxx, 1e-20)).reshape(1, *Sxx.shape)).to(device)
            mean, logstd = encoder.forward(encoder_input)
            assert mean.shape == (1, args.embedding_size)
            return np.array(mean.cpu()).reshape(-1)
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
    raw_dataset = dataloader.get_dataset(None, 1024)
    if args.classes is not None:
        v = list(raw_dataset.items())
        random.shuffle(v)
        raw_dataset = dict(v[:max(0, min(len(v), args.classes))])
    X_train, Y_train = prep_dataset({ k: v[:len(v) // 2] for k, v in raw_dataset.items() })
    X_eval, Y_eval = prep_dataset({ k: v[len(v) // 2:] for k, v in raw_dataset.items() })

    print(f'loaded {len(X_train)} training points and {len(X_eval)} eval points')
    clusters, cluster_labels = train_cluster_classifier(X_train, Y_train, n_clusters = 4000)
    print('trained cluster classifier:', clusters.shape, cluster_labels)
    print()

    confusion = test_cluster_classifier(clusters, cluster_labels, X_train, Y_train, len(raw_dataset))
    print(f'train accuracy: {np.einsum("ii", confusion) / len(X_train):.2f} ({round(np.einsum("ii", confusion))} / {len(X_train)})')
    print('confusion:')
    print_confusion_matrix(confusion, list(raw_dataset.keys()))
    print()

    confusion = test_cluster_classifier(clusters, cluster_labels, X_eval, Y_eval, len(raw_dataset))
    print(f'eval accuracy: {np.einsum("ii", confusion) / len(X_eval):.2f} ({round(np.einsum("ii", confusion))} / {len(X_eval)}) -- confusion:')
    print('confusion:')
    print_confusion_matrix(confusion, list(raw_dataset.keys()))
    print()
