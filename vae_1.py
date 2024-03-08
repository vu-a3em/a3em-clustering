import numpy as np
import torch
import argparse
import sys
import os

import dataloader

class Encoder(torch.nn.Module):
    def __init__(self, *, embedding_size):
        super().__init__()

        self.c1 = torch.nn.Conv2d(1, 256, kernel_size = (4, 4), stride = 2)
        self.c2 = torch.nn.Conv2d(256, 128, kernel_size = (4, 4), stride = 2)
        self.c3 = torch.nn.Conv2d(128, 64, kernel_size = (4, 4), stride = 2)

        self.d1 = torch.nn.Linear(8064, 1024)
        self.d2 = torch.nn.Linear(1024, 512)
        self.d3 = torch.nn.Linear(512, 256)
        self.d4 = torch.nn.Linear(256, 128)

        self.mean = torch.nn.Linear(128, embedding_size)
        self.logstd = torch.nn.Linear(128, embedding_size)
    def forward(self, x):
        assert x.shape[1:] == (129, 89)
        relu = torch.nn.LeakyReLU(0.01)

        x = x.reshape(x.shape[0], 1, 129, 89)
        x = relu(self.c1(x))
        x = relu(self.c2(x))
        x = relu(self.c3(x))

        x = x.reshape(x.shape[0], -1)
        x = relu(self.d1(x))
        x = relu(self.d2(x))
        x = relu(self.d3(x))
        x = relu(self.d4(x))

        return self.mean(x), self.logstd(x)

class Decoder(torch.nn.Module):
    def __init__(self, *, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.d1 = torch.nn.Linear(embedding_size, 128)
        self.d2 = torch.nn.Linear(128, 256)
        self.d3 = torch.nn.Linear(256, 512)
        self.d4 = torch.nn.Linear(512, 1024)
        self.d5 = torch.nn.Linear(1024, 8064)

        self.tc1 = torch.nn.ConvTranspose2d(64, 128, kernel_size = (4, 4), stride = 2)
        self.c1  = torch.nn.Conv2d(128, 128, kernel_size = (7, 7), padding = 3, groups = 128, padding_mode = 'replicate')
        self.c2  = torch.nn.Conv2d(128, 128, kernel_size = (5, 5), padding = 2, groups = 128, padding_mode = 'replicate')
        self.c3  = torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, groups = 128, padding_mode = 'replicate')
        self.tc2 = torch.nn.ConvTranspose2d(128, 256, kernel_size = (4, 4), stride = 2, output_padding = 1)
        self.c4  = torch.nn.Conv2d(256, 256, kernel_size = (7, 7), padding = 3, groups = 256, padding_mode = 'replicate')
        self.c5  = torch.nn.Conv2d(256, 256, kernel_size = (5, 5), padding = 2, groups = 256, padding_mode = 'replicate')
        self.c6  = torch.nn.Conv2d(256, 256, kernel_size = (3, 3), padding = 1, groups = 256, padding_mode = 'replicate')
        self.tc3 = torch.nn.ConvTranspose2d(256, 1, kernel_size = (4, 4), stride = 2, output_padding = 1)
        self.c7  = torch.nn.Conv2d(1, 1, kernel_size = (7, 7), padding = 3, groups = 1, padding_mode = 'replicate')
        self.c8  = torch.nn.Conv2d(1, 1, kernel_size = (5, 5), padding = 2, groups = 1, padding_mode = 'replicate')
        self.c9  = torch.nn.Conv2d(1, 1, kernel_size = (3, 3), padding = 1, groups = 1, padding_mode = 'replicate')
    def forward(self, x):
        assert x.shape[1:] == (self.embedding_size,)
        relu = torch.nn.LeakyReLU(0.01)

        x = relu(self.d1(x))
        x = relu(self.d2(x))
        x = relu(self.d3(x))
        x = relu(self.d4(x))
        x = relu(self.d5(x))

        x = x.reshape(x.shape[0], 64, 14, 9)
        x = relu(self.tc1(x))
        x = relu(self.c1(x))
        x = relu(self.c2(x))
        x = relu(self.c3(x))
        x = relu(self.tc2(x))
        x = relu(self.c4(x))
        x = relu(self.c5(x))
        x = relu(self.c6(x))
        x = relu(self.tc3(x))
        x = relu(self.c7(x))
        x = relu(self.c8(x))
        return self.c9(x).reshape(x.shape[0], 129, 89)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedding-size', type = int, required = True)
    parser.add_argument('--output', '-o', type = str, required = True)
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f'path \'{args.output}\' already exists', file = sys.stderr)
        sys.exit(1)
    os.mkdir(args.output)

    def prep_sample(x):
        f, t, Sxx = dataloader.get_spectrogram(x)
        return np.log10(np.maximum(Sxx, 1e-20))
    def prep_dataset(data):
        X = []
        Y = []
        for i, (label, samples) in enumerate(data.items()):
            for sample in samples:
                X.append(prep_sample(sample))
                Y.append(i)
        return np.array(X), np.array(Y)
    raw_dataset = dataloader.get_dataset(32, 1024)
    X_train, Y_train = prep_dataset({ k: v[:len(v) // 2] for k, v in raw_dataset.items() })
    X_eval, Y_eval = prep_dataset({ k: v[len(v) // 2:] for k, v in raw_dataset.items() })

    print(f'gpu enabled: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(embedding_size = args.embedding_size).to(device)
    decoder = Decoder(embedding_size = args.embedding_size).to(device)
    enc_X_train = X_train
    batch_size = 70
    batches_per_opt_step = 1
    lr = 2e-4
    weight_decay = 3e-4

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr = lr, weight_decay = weight_decay, amsgrad = True)
    def loss_fn(before, after, mean, logstd):
        mse_loss = torch.mean(torch.sum((before - after)**2, axis = 1))
        kldiv_loss = torch.mean(-0.5 * torch.sum(1 + logstd - logstd.exp()**2 - mean**2, axis = 1))
        return mse_loss + kldiv_loss

    def do_training_epoch(epoch):
        encoder.train(True)
        decoder.train(True)

        train_dataset = np.copy(X_train)
        np.random.shuffle(train_dataset)
        losses = []
        k = len(train_dataset) // batch_size
        for i in range(k):
            print(f'\repoch {epoch}: {i}/{k} complete ({100*i/k:.2f}%) ...', end = '')
            if i % batches_per_opt_step == 0:
                opt.zero_grad()
            before = torch.tensor(train_dataset[i * batch_size : (i + 1) * batch_size]).to(device)
            mean, logstd = encoder.forward(before)
            embed = mean + logstd.exp() * torch.normal(torch.zeros(logstd.shape), torch.ones(logstd.shape)).to(device)
            after = decoder.forward(embed)
            loss = loss_fn(before, after, mean, logstd)
            loss.backward()
            losses.append(loss.item())
            if (i + 1) % batches_per_opt_step == 0:
                opt.step()
        res = np.mean(losses)
        print(f'\repoch {epoch}: avg loss {res:.4f}                              ')
        return res
    def do_eval_epoch():
        encoder.train(False)
        decoder.train(False)
        with torch.no_grad():
            eval_dataset = np.copy(X_eval)
            np.random.shuffle(eval_dataset)
            losses = []
            k = len(eval_dataset) // batch_size
            for i in range(k):
                print(f'\reval: {i}/{k} complete ({100*i/k:.2f}%) ...', end = '')
                before = torch.tensor(eval_dataset[i * batch_size : (i + 1) * batch_size]).to(device)
                mean, logstd = encoder.forward(before)
                embed = mean + logstd.exp() * torch.normal(torch.zeros(logstd.shape), torch.ones(logstd.shape)).to(device)
                after = decoder.forward(embed)
                loss = loss_fn(before, after, mean, logstd)
                losses.append(loss.item())
            res = np.mean(losses)
            print(f'\reval: avg loss {res:.4f}                              ')
            return res

    with open(f'{args.output}/loss.csv', 'w') as log:
        log.write(f'epoch,train_loss,eval_loss\n')
        for i in range(256):
            train_loss = do_training_epoch(i)
            eval_loss = do_eval_epoch()
            log.write(f'{i},{train_loss},{eval_loss}\n')
            log.flush()
            if (i + 1) % 8 == 0:
                torch.save(encoder.state_dict(), f'{args.output}/encoder-E{i+1}-L{round(eval_loss)}.pt')
                torch.save(decoder.state_dict(), f'{args.output}/decoder-E{i+1}-L{round(eval_loss)}.pt')
                print('saved state dicts')
