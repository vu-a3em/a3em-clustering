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

Json files can also be piped into the simulation via the special token `-` as the input file:

```sh
cat my-sim.json | python sim.py -
```

Loading from stdin can be helpful when using the json generator `gen.py`.

```sh
python gen.py -h
```

To visualize a labeled output, you can use `vis.py`.

```sh
python vis.py -h
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

# Visualizer File Schema

The visualizer is used to create plots of a labeled output wav file.
The labeled outputs are given by a single JSON file containing an array of events.
Each event takes the form of an object with a `kind` field representing the label, as well as `start` and `duration` fields measured in seconds.
Below is an example labeled events file containing two events.

```json
[
    {
        "kind": "dog-bark",
        "start": 5.6495846184671,
        "duration": 0.428
    },
    {
        "kind": "cat-meow",
        "start": 7.90988560426518,
        "duration": 0.831
    }
]
```
