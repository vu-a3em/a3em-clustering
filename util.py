from pydub import AudioSegment

SOUND_CACHE = {}
def load_sound(path: str) -> AudioSegment:
    r = SOUND_CACHE.get(path, None)
    if r is None:
        r = SOUND_CACHE[path] = AudioSegment.from_file(path)
    return r
