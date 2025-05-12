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
    # [0, 'Frog:8', 'Rooster:1', 'Aircraft:1'],
    [1, 'Frog:1', 'Rooster:8', 'Aircraft:1'],
]
MAX_CLUSTERS = [
    # 8,
    # 16,
    # 24,
    # 32,
    # 40,
    # 48,
    # 56,
    # 64,
    # 96,
    112,
    128,
    144,
    # 160,
    # 192,
    # 256,
    # 300,
    # 320,
    # 384,
    # 512,
    # 1024,
]
MAX_WEIGHT = [
    1,
    8,
    16,
    32,
    40,
    48,
    56,
    64,
    # 80,
    # 96,
    # 128,
    # 192,
    # 256,
    # 512,
    # 1024,
]
FILTER_THRESH = [
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    # 0.006,
    # 0.007,
    # 0.008,
    # 0.009,
    # 0.010,
    # 0.020,
    # 0.050,
    # 0.100,
    # 0.200,
    # 0.300,
    # 0.400,
    # 0.500,
    # 0.600,
    # 0.700,
    # 0.800,
    # 0.900,
    # 1.000,
    # 1.100,
    # 1.200,
    # 1.3,
    # 1.4,
    # 1.5,
    # 1.6,
    # 1.7,
    # 1.8,
    # 1.9,
    # 2.0,
]
BACKGROUND_SCALE = [
    0.0,
    # 1.0,
]

# -------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type = int, required = True)
    parser.add_argument('--iterations', type = int, default = 1)
    parser.add_argument('--hours', type = float, required = True)
    parser.add_argument('--output', type = str, required = True)
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
                    'time', 'python', 'sim2-rmszc.py', '--quiet',
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
                ], stdout = f, stderr = f)
    workers = [threading.Thread(target = worker) for _ in range(args.jobs)]
    for w in workers: w.start()
    for w in workers: w.join()
