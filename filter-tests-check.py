import argparse
import numpy as np
import re
import os

DATA_REGEX = re.compile(r'^\s*(.*):\s*(\d+)\s*->\s*(\d+)')
def read_file(path: str) -> np.array:
    try:
        with open(path, 'r') as f:
            data = f.readlines()
        data = [DATA_REGEX.search(x) for x in data]
        data = {x.group(1): [int(x.group(2)), int(x.group(3))] for x in data if x is not None}
        del data['None']
        return np.array(list(data.values()))
    except Exception as e:
        print('error:', path)
        raise e

def metric(data: np.ndarray) -> float:
    ratios = data[:,1] / data[:,0]
    return np.mean(data[:,1])**0.9 * (np.min(ratios[1:]) - ratios[0]) / (ratios[0] if ratios[0] > 0 else 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type = str)
    parser.add_argument('-n', type = int, default = 3)
    args = parser.parse_args()

    data = [[x, metric(read_file(f'{args.path}/{x}'))] for x in os.listdir(args.path)]
    data.sort(key = lambda x: -x[1])
    for x in data[:min(len(data), args.n)]:
        print(x)
        with open(f'{args.path}/{x[0]}', 'r') as f: print(f.read())
        print()