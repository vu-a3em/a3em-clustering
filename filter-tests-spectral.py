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
    8,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    96,
    128,
    192,
    256,
    320,
    384,
    512,
    # 1024,
]
MAX_WEIGHT = [
    1,
    8,
    16,
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
    0.030,
    0.031,
    0.032,
    0.033,
    0.034,
    0.035,
    0.036,
    0.037,
    0.038,
    0.039,
    0.040,
    0.041,
    0.042,
    0.043,
    0.044,
    0.045,
    0.046,
    0.047,
    0.048,
    0.049,
    0.050,
    0.051,
    0.052,
    0.053,
    0.054,
    0.055,
    0.056,
    0.057,
    0.058,
    0.059,
    0.060,
    0.061,
    0.062,
    0.063,
    0.064,
    0.065,
    0.066,
    0.067,
    0.068,
    0.069,
    0.070,
    # 0.08,
    # 0.09,
    # 0.10,
    # 0.11,
    # 0.12,
    # 0.13,
    # 0.14,
    # 0.15,
    # 0.20,
    # 0.30,
    # 0.40,
    # 0.50,
    # 0.60,
    # 0.70,
    # 0.80,
    # 0.90,
    # 1.00,
    # 1.10,
    # 1.20,
    # 1.30,
    # 1.40,
    # 1.50,
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
                ], stdout = f, stderr = f)
    workers = [threading.Thread(target = worker) for _ in range(args.jobs)]
    for w in workers: w.start()
    for w in workers: w.join()
