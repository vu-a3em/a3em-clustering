# Installation

First, install the necessary python dependencies:

```sh
pip install pydub
```

# Usage

Simply run `sim.py` with the path to a simulation file:

```sh
python sim.py my-sim.json
```

# Simulation File Schema

Each simulation is a single JSON file.
All audio files to be included are events (including background noise).
Each event takes a source path (any file format supported by `ffmpeg`), a start time (in seconds), and a list of zero or more effects to apply sequentially.
The entire simulation can also have effects applied to it after combining.

```json
{
    "events": [
        {
            "source": "my/background/noise.wav",
            "start": 0,
            "effects": []
        },
        {
            "source": "my/sound/event.mp3",
            "start": 12,
            "effects": [
                ["gain", 6]
            ]
        }
    ],
    "effects": []
}
```

Note: `start` and `effects` can be omitted, in which case they default to `0` and `[]`, respectively.
Each event can also contain extra metadata (e.g., a class label) for other tools to use, which are ignored by `sim.py`.

## Effects - Gain

Applies a gain/volume effect, specified in dB.

```json
["gain", <amount>]
```

## Effects - Slice

Slices the audio clip at two points specified in seconds.

```json
["slice", <start>, <stop>]
```

## Effects - mp3

Round trips the clip to mp3 and back into (in-memory) wav.

```json
["mp3"]
```
