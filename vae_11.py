import numpy as np
import torch
import argparse
import sys
import os

import dataloader

PADDING_MODE = 'replicate'

class Encoder(torch.nn.Module):
    def __init__(self, *, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.convolutions = torch.nn.ModuleList([
            torch.nn.Conv2d(1, 64, kernel_size = (3, 3), stride = (2, 1)),
            torch.nn.Conv2d(64, 128, kernel_size = (1, 1)),
            torch.nn.Conv2d(128, 64, kernel_size = (1, 1)),
            torch.nn.Conv2d(64, 32, kernel_size = (1, 1)),

            torch.nn.Conv2d(32, 64, kernel_size = (3, 3), stride = (2, 1)),
            torch.nn.Conv2d(64, 128, kernel_size = (1, 1)),
            torch.nn.Conv2d(128, 64, kernel_size = (1, 1)),
            torch.nn.Conv2d(64, 32, kernel_size = (1, 1)),

            torch.nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 2),
            torch.nn.Conv2d(64, 128, kernel_size = (1, 1)),
            torch.nn.Conv2d(128, 64, kernel_size = (1, 1)),
            torch.nn.Conv2d(64, 32, kernel_size = (1, 1)),

            torch.nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 2),
            torch.nn.Conv2d(64, 128, kernel_size = (1, 1)),
            torch.nn.Conv2d(128, 64, kernel_size = (1, 1)),
            torch.nn.Conv2d(64, 32, kernel_size = (1, 1)),

            torch.nn.Conv2d(32, 64, kernel_size = (3, 3), stride = 2),
            torch.nn.Conv2d(64, 128, kernel_size = (1, 1)),
            torch.nn.Conv2d(128, 64, kernel_size = (1, 1)),
            torch.nn.Conv2d(64, 32, kernel_size = (1, 1)),
        ])
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(288, 128),
            torch.nn.Linear(128, 64),
        ])
        self.mean = torch.nn.Linear(64, embedding_size)
        self.logstd = torch.nn.Linear(64, embedding_size)
    def forward(self, x):
        assert x.shape[1:] == (129, 35), f'{x.shape}'
        relu = torch.nn.LeakyReLU(0.01)

        x = x.reshape(x.shape[0], 1, 129, 35)
        for step in self.convolutions:
            x = relu(step(x))
            # print(f'>>> {x.shape}')
        x = x.reshape(x.shape[0], -1)
        for step in self.linears:
            x = relu(step(x))

        return self.mean(x), self.logstd(x)

class Decoder(torch.nn.Module):
    def __init__(self, *, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(embedding_size, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 576),
        ])
        self.convolutions = torch.nn.ModuleList([
            torch.nn.ConvTranspose2d(64, 128, kernel_size = (3, 3), stride = 2),
            torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, padding_mode = PADDING_MODE),
            torch.nn.Conv2d(128, 128, kernel_size = (1, 1)),

            torch.nn.ConvTranspose2d(128, 128, kernel_size = (3, 3), stride = 2),
            torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, padding_mode = PADDING_MODE),
            torch.nn.Conv2d(128, 128, kernel_size = (1, 1)),

            torch.nn.ConvTranspose2d(128, 64, kernel_size = (3, 3), stride = 2),
            torch.nn.Conv2d(64, 64, kernel_size = (3, 3), padding = 1, padding_mode = PADDING_MODE),
            torch.nn.Conv2d(64, 64, kernel_size = (1, 1)),

            torch.nn.ConvTranspose2d(64, 64, kernel_size = (3, 3), stride = (2, 1), output_padding = (1, 0)),
            torch.nn.Conv2d(64, 64, kernel_size = (3, 3), padding = 1, padding_mode = PADDING_MODE),
            torch.nn.Conv2d(64, 64, kernel_size = (1, 1)),

            torch.nn.ConvTranspose2d(64, 32, kernel_size = (3, 3), stride = (2, 1)),
            torch.nn.Conv2d(32, 32, kernel_size = (5, 5), padding = 2, padding_mode = PADDING_MODE),
            torch.nn.Conv2d(32, 16, kernel_size = (1, 1)),
            torch.nn.Conv2d(16, 1, kernel_size = (1, 1)),
        ])
    def forward(self, x):
        assert x.shape[1:] == (self.embedding_size,)
        relu = torch.nn.LeakyReLU(0.01)

        for step in self.linears:
            x = relu(step(x))
        x = x.reshape(x.shape[0], 64, 3, 3)
        for i, step in enumerate(self.convolutions):
            if i == len(self.convolutions) - 1:
                return step(x).reshape(x.shape[0], 129, 35)
            x = relu(step(x))
            # print(f'< {x.shape}')
        raise RuntimeError('unreachable!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-size', '-e', type = int, required = True)
    parser.add_argument('--output', '-o', type = str, required = True)
    parser.add_argument('--alpha', '-a', type = str, required = True)
    args = parser.parse_args()

    print(f'gpu enabled: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    alphas = np.array([float(x) for x in args.alpha.split(':')])
    assert alphas.shape == (3,) and np.all(alphas >= 0) and np.sum(alphas) > 0

    encoder = Encoder(embedding_size = args.embedding_size).to(device)
    decoder = Decoder(embedding_size = args.embedding_size).to(device)
    s = dataloader.get_spectrogram_normalized(next(iter(dataloader.get_dataset(1, 1).items()))[1][0])[0].shape
    ss = decoder.forward(encoder.forward(torch.rand((1, *s)).to(device))[0]).shape[1:]
    assert ss == s, f'{s} -> {ss}'

    if os.path.exists(args.output):
        print(f'path \'{args.output}\' already exists', file = sys.stderr)
        sys.exit(1)
    os.mkdir(args.output)

    def prep_dataset(data):
        X = []
        Y = []
        for i, (label, samples) in enumerate(data.items()):
            for sample in samples:
                X.append(dataloader.get_spectrogram_normalized(sample)[0])
                Y.append(i)
        return np.array(X, dtype = np.float32), np.array(Y, dtype = np.float32)
    raw_dataset = dataloader.get_dataset(None, 8192)
    # raw_dataset = dataloader.get_dataset(5, 50)
    X_train, Y_train = prep_dataset({ k: v[:len(v) // 2] for k, v in raw_dataset.items() })
    X_eval, Y_eval = prep_dataset({ k: v[len(v) // 2:] for k, v in raw_dataset.items() })

    enc_X_train = X_train
    batch_size = 35
    batches_per_opt_step = 1
    lr = 3e-4
    weight_decay = 1.5e-5
    epochs = 256
    current_epoch = [0]
    energy_factor = 1e-7

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = weight_decay, amsgrad = True)
    def loss_fn(before, after, mean, logstd):
        if torch.any(torch.isnan(mean)):
            raise RuntimeError('ugh')

        mse_loss = torch.mean(torch.sum((before - after)**2, axis = (1, 2)), axis = 0)
        kldiv_loss = torch.mean(-0.5 * torch.sum(1 + 2 * logstd - logstd.exp()**2 - mean**2, axis = 1), axis = 0)

        embed_dists = torch.stack([torch.sum((mean - mean[i])**2, axis = 1) for i in range(before.shape[0])])
        energies = torch.sum(before**2, axis = 2) # integrate out time
        delta_energies = energy_factor * torch.stack([torch.sum((energies - energies[i])**2, axis = 1) for i in range(before.shape[0])])
        sim_loss = torch.mean((embed_dists - delta_energies)**2, axis = (0, 1))

        a = current_epoch[0] / epochs
        m = alphas * np.array([1, a, a])
        return m[0] * mse_loss + m[1] * kldiv_loss + m[2] * sim_loss, mse_loss, kldiv_loss, sim_loss

    def do_training_epoch():
        encoder.train()
        decoder.train()

        train_dataset = np.copy(X_train)
        np.random.shuffle(train_dataset)
        losses = []
        k = len(train_dataset) // batch_size
        for i in range(k):
            if i % batches_per_opt_step == 0:
                opt.zero_grad()
            before = torch.tensor(train_dataset[i * batch_size : (i + 1) * batch_size]).to(device)
            mean, logstd = encoder.forward(before)
            embed = mean + logstd.exp() * torch.normal(torch.zeros(logstd.shape), torch.ones(logstd.shape)).to(device)
            after = decoder.forward(embed)
            loss, mse_loss, kldiv_loss, sim_loss = loss_fn(before, after, mean, logstd)
            loss.backward()
            losses.append([ loss.item(), mse_loss.item(), kldiv_loss.item(), sim_loss.item() ])
            if (i + 1) % batches_per_opt_step == 0:
                opt.step()
            print(f'\repoch {current_epoch[0]}: {i+1}/{k} complete ({100*(i+1)/k:.2f}%) ... batch loss {loss.item():.2f} (mse {mse_loss.item():.2f}) (kldiv {kldiv_loss.item():.2f}) (sim {sim_loss.item():.2f})          ', end = '')
        res = np.mean(losses, axis = 0)
        print(f'\repoch {current_epoch[0]}: avg loss {res[0]:.4f} (mse {res[1]:.4f}) (kldiv {res[2]:.4f}) (sim {res[3]:.4f})                                                                                              ')
        return res
    def do_eval_epoch():
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            eval_dataset = np.copy(X_eval)
            np.random.shuffle(eval_dataset)
            losses = []
            k = len(eval_dataset) // batch_size
            for i in range(k):
                before = torch.tensor(eval_dataset[i * batch_size : (i + 1) * batch_size]).to(device)
                mean, logstd = encoder.forward(before)
                embed = mean + logstd.exp() * torch.normal(torch.zeros(logstd.shape), torch.ones(logstd.shape)).to(device)
                after = decoder.forward(embed)
                loss, mse_loss, kldiv_loss, sim_loss = loss_fn(before, after, mean, logstd)
                losses.append([ loss.item(), mse_loss.item(), kldiv_loss.item(), sim_loss.item() ])
                print(f'\reval: {i+1}/{k} complete ({100*(i+1)/k:.2f}%) ... batch loss {loss.item():.2f} (mse {mse_loss.item():.2f}) (kldiv {kldiv_loss.item():.2f}) (sim {sim_loss.item():.2f})          ', end = '')
            res = np.mean(losses, axis = 0)
            print(f'\reval: avg loss {res[0]:.4f} (mse {res[1]:.4f}) (kldiv {res[2]:.4f}) (sim {res[3]:.4f})                                                                                              ')
            return res

    with open(f'{args.output}/loss.csv', 'w') as log:
        log.write(f'epoch,train_loss,eval_loss\n')
        for i in range(epochs):
            current_epoch[0] = i
            train_loss = do_training_epoch()
            eval_loss = do_eval_epoch()
            log.write(f'{i},{train_loss[0]},{eval_loss[0]},{train_loss[1]},{eval_loss[1]},{train_loss[2]},{eval_loss[2]},{train_loss[3]},{eval_loss[3]}\n')
            log.flush()
            if (i + 1) % 8 == 0:
                torch.save(encoder.state_dict(), f'{args.output}/encoder-F{args.embedding_size}-A{":".join(str(x) for x in alphas)}-E{i+1}-L{round(eval_loss[0])}.pt')
                torch.save(decoder.state_dict(), f'{args.output}/decoder-F{args.embedding_size}-A{":".join(str(x) for x in alphas)}-E{i+1}-L{round(eval_loss[0])}.pt')
                print('saved state dicts')
