import argparse
import numpy as np
from typing import Optional
import re
import os
import sys

DATA_REGEX = re.compile(r'^\s*(.*):\s*(\d+)\s*->\s*(\d+)')
def read_file(path: str) -> Optional[np.ndarray]:
    try:
        with open(path, 'r') as f:
            data = f.readlines()
        data = [DATA_REGEX.search(x) for x in data]
        data = {x.group(1): [int(x.group(2)), int(x.group(3))] for x in data if x is not None}
        del data['None']
        return np.array(list(data.values()))
    except Exception as e:
        return None

def metric(data: np.ndarray) -> float:
    ratios = data[:,1] / data[:,0]
    return np.mean(data[:,1])**0.92 * (np.min(ratios[1:]) - ratios[0]) / (ratios[0] if ratios[0] > 0 else 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type = str)
    parser.add_argument('-n', type = int, default = 3)
    parser.add_argument('--purge', action = 'store_true')
    args = parser.parse_args()

    data = []
    for x in os.listdir(args.path):
        p = f'{args.path}/{x}'
        y = read_file(p)
        if y is not None:
            data.append([x, metric(y)])
        elif args.purge:
            os.remove(p)
        else:
            print(f'corrupted data: {p}', file = sys.stderr)
            sys.exit(1)

    data = [[x, metric(read_file(f'{args.path}/{x}'))] for x in os.listdir(args.path)]
    data.sort(key = lambda x: -x[1])
    for x in data[:min(len(data), args.n)]:
        print(x)
        with open(f'{args.path}/{x[0]}', 'r') as f: print(f.read())
        print()