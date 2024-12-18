import argparse
import random
import json
import sys
import os

from util import load_sound
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument('backgrounds')
parser.add_argument('events')
parser.add_argument('-n', '--number', type = int, required = True)
parser.add_argument('-w', '--weights', type = str, default = '_:1')
parser.add_argument('--min-gain', type = float, default = 0.0)
parser.add_argument('--max-gain', type = float, default = 0.0)
parser.add_argument('--mp3-chance', type = float, default = 0.0)
parser.add_argument('--seed', type = int)
args = parser.parse_args()

if args.seed:
    random.seed(args.seed)

def crash(msg):
    print(msg, file = sys.stderr)
    sys.exit(1)

def get_all_sounds(path: str) -> List[str]:
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            res.append(os.path.join(root, file))
    res.sort() # ensure a consistent ordering
    return res
def gen_effects() -> List:
    res = []

    gain = args.min_gain + random.random() * (args.max_gain - args.min_gain)
    if gain != 0:
        res.append(['gain', gain])

    if random.random() < args.mp3_chance:
        res.append(['mp3'])

    return res

backgrounds = get_all_sounds(args.backgrounds)
events = { k: get_all_sounds(f'{args.events}/{k}') for k in os.listdir(args.events) }
if len(backgrounds) == 0: crash('no background sounds')
for k, v in events.items():
    if len(v) == 0: crash(f'no sounds for event "{k}"')

weights = {}
weights_wildcard = None
for x in args.weights.split(','):
    k, v = x.split(':')
    v = float(v)
    if v <= 0: continue

    if k == '_':
        if weights_wildcard is not None: crash('multiple wildcards detected')
        weights_wildcard = v
    else:
        if k not in events: crash(f'unknown event class "{k}"')
        if k in weights: crash(f'multiple weights for class "{k}"')
        weights[k] = v
if weights_wildcard is not None:
    missing = set(events.keys()) - set(weights.keys())
    for x in missing:
        weights[x] = weights_wildcard / len(missing)
if len(weights) == 0: crash('no included event classes')
weights_total = sum(weights.values())
for k, v in weights.items():
    weights[k] = weights[k] / weights_total

def random_event_class() -> str:
    r = random.random()
    t = 0
    last_k = None
    for k, v in weights.items():
        last_k = k
        t += v
        if r <= t: break
    return last_k

background = random.choice(backgrounds)
background_length = load_sound(background).duration_seconds
output = {
    'events': [
        {
            'class': None,
            'source': background,
            'start': 0,
            'duration': background_length,
            'effects': gen_effects(),
        }
    ],
    'effects': [],
}
for i in range(args.number):
    event_class = random_event_class()
    event = random.choice(events[event_class])
    event_length = load_sound(event).duration_seconds
    if event_length > background_length:
        print(f'event "{event}" is longer than background "{background}" ({event_length} s vs {background_length} s)', file = sys.stderr)
        sys.exit(1)
    t = random.random() * (background_length - event_length)
    output['events'].append({
        'class': event_class,
        'source': event,
        'start': t,
        'duration': event_length,
        'effects': gen_effects(),
    })
json.dump(output, sys.stdout, indent = 4)
print()
