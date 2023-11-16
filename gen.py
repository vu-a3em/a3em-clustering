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
parser.add_argument('-n', type = int, required = True)
parser.add_argument('--min-gain', type = float, default = 0.0)
parser.add_argument('--max-gain', type = float, default = 0.0)
parser.add_argument('--mp3-chance', type = float, default = 0.0)
parser.add_argument('--seed', type = int)
args = parser.parse_args()

if args.seed:
    random.seed(args.seed)

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
events = get_all_sounds(args.events)
for k, v in { 'backgrounds': backgrounds, 'events': events }.items():
    if len(v) == 0:
        print(f'no {k} found!', file = sys.stderr)
        sys.exit(1)

background = random.choice(backgrounds)
background_length = load_sound(background).duration_seconds
output = {
    'events': [
        {
            'source': background,
            'start': 0,
            'duration': background_length,
            'effects': gen_effects(),
        }
    ],
    'effects': [],
}
for i in range(args.n):
    event = random.choice(events)
    event_length = load_sound(event).duration_seconds
    if event_length > background_length:
        print(f'event "{event}" is longer than background "{background}" ({event_length} s vs {background_length} s)', file = sys.stderr)
        sys.exit(1)
    t = random.random() * (background_length - event_length)
    output['events'].append({
        'source': event,
        'start': t,
        'duration': event_length,
        'effects': gen_effects(),
    })
json.dump(output, sys.stdout, indent = 4)
print()
