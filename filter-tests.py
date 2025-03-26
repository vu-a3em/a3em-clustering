import subprocess
import argparse
import itertools
import threading
import os

EVENTS = [
    '/home/devin/Downloads/dog-events/',
    '/home/devin/Downloads/misc-events/',
]
BACKGROUNDS = [
    '/home/devin/Downloads/edansa-events/',
]

# -------------------------------------------------------

EVENT_FREQS = [
    [0, 'Crow:8', 'Rooster:1', 'Aircraft:1'],
]
MAX_CLUSTERS = [
    64, 128, 256, 512, 1024,
]
MAX_WEIGHT = [
    4, 16, 64,
]
FILTER_THRESH = [
    0.3, 0.35, 0.4, 0.45, 0.5,
]
VOTE_THRESH = [
    0.0, 0.05, 0.1, 0.15, 0.2,
]

# -------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type = int, required = True)
    parser.add_argument('--hours', type = float, required = True)
    parser.add_argument('--output', type = str, required = True)
    args = parser.parse_args()

    assert args.jobs >= 1
    assert args.hours >= 0.1

    os.mkdir(args.output)

    queue = list(itertools.product(EVENT_FREQS, MAX_CLUSTERS, MAX_WEIGHT, FILTER_THRESH, VOTE_THRESH))
    queue_init_size = len(queue)
    queue_lock = threading.Lock()
    def worker():
        while True:
            with queue_lock:
                if len(queue) == 0: return
                i = queue_init_size - len(queue)
                info = queue.pop()
                print(f'starting task {i + 1}/{queue_init_size}')
            with open(f'{args.output}/v{info[4]}-c{info[1]}-w{info[2]}-t{info[3]}-e{info[0][0]}.txt', 'w') as f:
                subprocess.check_call([
                    'time', 'python', 'sim2.py', '--quiet',
                    '--events', *EVENTS,
                    '--backgrounds', *BACKGROUNDS,
                    '--clips', str(round(2 * 60 * args.hours)),
                    '--event-prob', '0.75',
                    '--event-freqs', *info[0][1:],
                    '--max-clusters', str(info[1]),
                    '--max-weight', str(info[2]),
                    '--filter-thresh', str(info[3]),
                    '--vote-thresh', str(info[4]),
                ], stdout = f, stderr = f)
    workers = [threading.Thread(target = worker) for _ in range(args.jobs)]
    for w in workers: w.start()
    for w in workers: w.join()
