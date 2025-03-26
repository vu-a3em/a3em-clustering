import argparse
import numpy as np
import re
import os

DATA_REGEX = re.compile(r'^\s*(.*):\s*(\d+)\s*->\s*(\d+)')
def read_file(path: str) -> np.array:
    with open(path, 'r') as f:
        data = f.readlines()
    data = [DATA_REGEX.search(x) for x in data]
    data = {x.group(1): [int(x.group(2)), int(x.group(3))] for x in data if x is not None}
    del data['None']
    return np.array(list(data.values()))

def metric(data: np.ndarray) -> float:
    ratios = data[:,1] / data[:,0]
    return np.log(1 + np.mean(data[:,1])) * (np.min(ratios[1:]) - ratios[0]) / ratios[0]
    # return np.mean(data[:,1]) / (1 + np.max(data[:,1]) - np.min(data[:,1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type = str)
    args = parser.parse_args()

    data = [[x, metric(read_file(f'{args.path}/{x}'))] for x in os.listdir(args.path)]
    data.sort(key = lambda x: -x[1])
    for x in data[:5]:
        print(x)
        with open(f'{args.path}/{x[0]}', 'r') as f: print(f.read())
        print()