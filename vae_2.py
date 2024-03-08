import numpy as np
import torch
import argparse
import sys
import os

import dataloader

PADDING_MODE = 'replicate'
DEPTH_SEPARABLE = False

def get_groups(k):
    return k if DEPTH_SEPARABLE else 1

class Encoder(torch.nn.Module):
    def __init__(self, *, embedding_size):
        super().__init__()

        self.mc1 = torch.nn.Conv2d(1, 128, kernel_size = (4, 4), stride = 2)
        self.c11 = torch.nn.Conv2d(128, 128, kernel_size = (7, 7), padding = 3, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.c12 = torch.nn.Conv2d(128, 128, kernel_size = (5, 5), padding = 2, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.c13 = torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, groups = get_groups(128), padding_mode = PADDING_MODE)

        self.mc2 = torch.nn.Conv2d(128, 256, kernel_size = (4, 4), stride = 2)
        self.c21 = torch.nn.Conv2d(256, 256, kernel_size = (7, 7), padding = 3, groups = get_groups(256), padding_mode = PADDING_MODE)
        self.c22 = torch.nn.Conv2d(256, 256, kernel_size = (5, 5), padding = 2, groups = get_groups(256), padding_mode = PADDING_MODE)
        self.c23 = torch.nn.Conv2d(256, 256, kernel_size = (3, 3), padding = 1, groups = get_groups(256), padding_mode = PADDING_MODE)

        self.mc3 = torch.nn.Conv2d(256, 128, kernel_size = (4, 4), stride = 2)
        self.c31 = torch.nn.Conv2d(128, 128, kernel_size = (7, 7), padding = 3, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.c32 = torch.nn.Conv2d(128, 128, kernel_size = (5, 5), padding = 2, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.c33 = torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, groups = get_groups(128), padding_mode = PADDING_MODE)

        self.mc4 = torch.nn.Conv2d(128, 64, kernel_size = (4, 4), stride = 2)
        self.c41 = torch.nn.Conv2d(64, 64, kernel_size = (7, 7), padding = 3, groups = get_groups(64), padding_mode = PADDING_MODE)
        self.c42 = torch.nn.Conv2d(64, 64, kernel_size = (5, 5), padding = 2, groups = get_groups(64), padding_mode = PADDING_MODE)
        self.c43 = torch.nn.Conv2d(64, 64, kernel_size = (3, 3), padding = 1, groups = get_groups(64), padding_mode = PADDING_MODE)

        self.d1 = torch.nn.Linear(1152, 512)
        self.d2 = torch.nn.Linear(512, 256)
        self.d3 = torch.nn.Linear(256, 128)

        self.mean = torch.nn.Linear(128, embedding_size)
        self.logstd = torch.nn.Linear(128, embedding_size)
    def forward(self, x):
        assert x.shape[1:] == (129, 89)
        relu = torch.nn.LeakyReLU(0.01)

        x = x.reshape(x.shape[0], 1, 129, 89)

        x = relu(self.mc1(x))
        x = relu(self.c11(x))
        x = relu(self.c12(x))
        x = relu(self.c13(x))

        x = relu(self.mc2(x))
        x = relu(self.c21(x))
        x = relu(self.c22(x))
        x = relu(self.c23(x))

        x = relu(self.mc3(x))
        x = relu(self.c31(x))
        x = relu(self.c32(x))
        x = relu(self.c33(x))

        x = relu(self.mc4(x))
        x = relu(self.c41(x))
        x = relu(self.c42(x))
        x = relu(self.c43(x))

        x = x.reshape(x.shape[0], -1)

        x = relu(self.d1(x))
        x = relu(self.d2(x))
        x = relu(self.d3(x))

        return self.mean(x), self.logstd(x)

class Decoder(torch.nn.Module):
    def __init__(self, *, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.d1 = torch.nn.Linear(embedding_size, 128)
        self.d2 = torch.nn.Linear(128, 256)
        self.d3 = torch.nn.Linear(256, 512)
        self.d4 = torch.nn.Linear(512, 1152)

        self.tc1 = torch.nn.ConvTranspose2d(64, 128, kernel_size = (4, 4), stride = 2, output_padding = (0, 1))
        self.c11 = torch.nn.Conv2d(128, 128, kernel_size = (7, 7), padding = 3, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.c12 = torch.nn.Conv2d(128, 128, kernel_size = (5, 5), padding = 2, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.c13 = torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, groups = get_groups(128), padding_mode = PADDING_MODE)
        self.tc2 = torch.nn.ConvTranspose2d(128, 256, kernel_size = (4, 4), stride = 2, output_padding = (0, 0))
        self.c21 = torch.nn.Conv2d(256, 256, kernel_size = (7, 7), padding = 3, groups = get_groups(256), padding_mode = PADDING_MODE)
        self.c22 = torch.nn.Conv2d(256, 256, kernel_size = (5, 5), padding = 2, groups = get_groups(256), padding_mode = PADDING_MODE)
        self.c23 = torch.nn.Conv2d(256, 256, kernel_size = (3, 3), padding = 1, groups = get_groups(256), padding_mode = PADDING_MODE)
        self.tc3 = torch.nn.ConvTranspose2d(256, 128, kernel_size = (4, 4), stride = 2, output_padding = (1, 1))
        self.c31 = torch.nn.Conv2d(128, 128, kernel_size = (7, 7), padding = 3, groups = get_groups(1), padding_mode = PADDING_MODE)
        self.c32 = torch.nn.Conv2d(128, 128, kernel_size = (5, 5), padding = 2, groups = get_groups(1), padding_mode = PADDING_MODE)
        self.c33 = torch.nn.Conv2d(128, 128, kernel_size = (3, 3), padding = 1, groups = get_groups(1), padding_mode = PADDING_MODE)
        self.tc4 = torch.nn.ConvTranspose2d(128, 1, kernel_size = (4, 4), stride = 2, output_padding = (1, 1))
        self.c41 = torch.nn.Conv2d(1, 1, kernel_size = (7, 7), padding = 3, groups = get_groups(1), padding_mode = PADDING_MODE)
        self.c42 = torch.nn.Conv2d(1, 1, kernel_size = (5, 5), padding = 2, groups = get_groups(1), padding_mode = PADDING_MODE)
        self.c43 = torch.nn.Conv2d(1, 1, kernel_size = (3, 3), padding = 1, groups = get_groups(1), padding_mode = PADDING_MODE)
    def forward(self, x):
        assert x.shape[1:] == (self.embedding_size,)
        relu = torch.nn.LeakyReLU(0.01)

        x = relu(self.d1(x))
        x = relu(self.d2(x))
        x = relu(self.d3(x))
        x = relu(self.d4(x))

        x = x.reshape(x.shape[0], 64, 6, 3)
        x = relu(self.tc1(x))
        x = relu(self.c11(x))
        x = relu(self.c12(x))
        x = relu(self.c13(x))
        x = relu(self.tc2(x))
        x = relu(self.c21(x))
        x = relu(self.c22(x))
        x = relu(self.c23(x))
        x = relu(self.tc3(x))
        x = relu(self.c31(x))
        x = relu(self.c32(x))
        x = relu(self.c33(x))
        x = relu(self.tc4(x))
        x = relu(self.c41(x))
        x = relu(self.c42(x))
        return self.c43(x).reshape(x.shape[0], 129, 89)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedding-size', type = int, required = True)
    parser.add_argument('--output', '-o', type = str, required = True)
    args = parser.parse_args()

    print(f'gpu enabled: {torch.cuda.is_available()}')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    encoder = Encoder(embedding_size = args.embedding_size).to(device)
    decoder = Decoder(embedding_size = args.embedding_size).to(device)
    decoder.forward(encoder.forward(torch.rand((1, 129, 89)).to(device))[0])

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

    enc_X_train = X_train
    batch_size = 70
    batches_per_opt_step = 1
    lr = 1e-4
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
            print(f'\repoch {epoch}: {i+1}/{k} complete ({100*(i+1)/k:.2f}%) ... batch loss {loss.item():.2f}', end = '')
        res = np.mean(losses)
        print(f'\repoch {epoch}: avg loss {res:.4f}                                                                    ')
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
                before = torch.tensor(eval_dataset[i * batch_size : (i + 1) * batch_size]).to(device)
                mean, logstd = encoder.forward(before)
                embed = mean + logstd.exp() * torch.normal(torch.zeros(logstd.shape), torch.ones(logstd.shape)).to(device)
                after = decoder.forward(embed)
                loss = loss_fn(before, after, mean, logstd)
                losses.append(loss.item())
                print(f'\reval: {i+1}/{k} complete ({100*(i+1)/k:.2f}%) ... batch loss {loss.item():.2f}', end = '')
            res = np.mean(losses)
            print(f'\reval: avg loss {res:.4f}                                                                    ')
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
