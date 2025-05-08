import subprocess
import argparse
import itertools
import threading
import os

EVENTS = [
    # '/home/devin/Downloads/dog-events/',
    '/home/devin/Downloads/misc-events/',
]
BACKGROUNDS = [
    '/home/devin/Downloads/edansa-events/',
]

# -------------------------------------------------------

EVENT_FREQS = [
    [0, 'Frog:8', 'Rooster:1', 'Aircraft:1'],
    # [1, 'Frog:1', 'Rooster:8', 'Aircraft:1'],
]
MAX_CLUSTERS = [
    # 8,
    # 16,
    # 24,
    32,
    # 40,
    # 48,
    # 56,
    64,
    128,
    256,
    512,
    1024,
]
MAX_WEIGHT = [
    1,
    8,
    # 16,
    # 32,
    # 40,
    # 48,
    # 56,
    # 64,
    # 80,
    # 96,
    # 128,
    # 192,
    # 256,
    # 512,
    # 1024,
]
FILTER_THRESH = [
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.13,
    0.14,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
    1.10,
    1.20,
    1.30,
    1.40,
    1.50,
    # 5,
    # 10,
    # 100,
    # 500,
    # 1000,
]
BACKGROUND_SCALE = [
    # 0.0,
    1.0,
]

# -------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type = int, required = True)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--hours', type = float, required = True)
    parser.add_argument('--output', type = str, required = True)
    parser.add_argument('--quantized', action = 'store_true')
    args = parser.parse_args()

    assert args.jobs >= 1
    assert args.hours >= 0.1

    if not os.path.exists(args.output): os.mkdir(args.output)

    queue = list(itertools.product(EVENT_FREQS, MAX_CLUSTERS, MAX_WEIGHT, FILTER_THRESH, BACKGROUND_SCALE))
    queue_init_size = len(queue)
    queue_lock = threading.Lock()
    def worker():
        while True:
            with queue_lock:
                if len(queue) == 0: return
                i = queue_init_size - len(queue)
                info = queue.pop()
                print(f'starting task {i + 1}/{queue_init_size}')
            file = f'{args.output}/b{info[4]}-c{info[1]}-w{info[2]}-t{info[3]}-e{info[0][0]}.txt'
            if os.path.exists(file): continue
            with open(file, 'w') as f:
                subprocess.check_call([
                    'time', 'python', 'sim2-spectral.py', '--quiet',
                    '--events', *EVENTS,
                    '--backgrounds', *BACKGROUNDS,
                    '--clips', str(round(2 * 60 * args.hours)),
                    '--iterations', str(args.iterations),
                    '--event-prob', '1.0',
                    '--event-freqs', *info[0][1:],
                    '--max-clusters', str(info[1]),
                    '--max-weight', str(info[2]),
                    '--filter-thresh', str(info[3]),
                    '--background-scale', str(info[4]),
                    *(['--quantized'] if args.quantized else []),
                ], stdout = f, stderr = f)
    workers = [threading.Thread(target = worker) for _ in range(args.jobs)]
    for w in workers: w.start()
    for w in workers: w.join()
