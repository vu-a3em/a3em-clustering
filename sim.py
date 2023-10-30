from pydub import AudioSegment
from typing import List, Any
import json
import sys

if len(sys.argv) != 2:
    print(f'usage: {sys.argv[0]} [sim file]', file = sys.stderr)
    sys.exit(1)
with open(sys.argv[1], 'r') as f:
    sim = json.load(f)

def load_sound(path: str) -> AudioSegment:
    return AudioSegment.from_file(path)
def apply_effects(sound: AudioSegment, effects: List[Any]) -> AudioSegment:
    for effect in effects:
        if effect[0] == 'gain':
            sound = sound + effect[1]
        elif effect[0] == 'slice':
            sound = sound[effect[1] * 1000 : effect[2] * 1000]
        else:
            raise RuntimeError(f'unknown effect type: {effect[0]}')
    return sound

duration = 0
sounds = []
for event in sim['events']:
    sound = load_sound(event['source'])
    sounds.append(sound)
    duration = max(duration, event.get('start', 0) + sound.duration_seconds)

res = AudioSegment.silent(duration * 1000)
for i, event in enumerate(sim['events']):
    sound = apply_effects(sounds[i], event.get('effects', []))
    res = res.overlay(sound, event.get('start', 0) * 1000)
res = apply_effects(res, sim.get('effects', []))
res.export('out.wav')
