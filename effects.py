import soundfile
import librosa
import numpy as np
import io

def mp3_roundtrip(wav: np.ndarray, sr: int) -> np.ndarray:
    b = io.BytesIO()
    soundfile.write(b, librosa.resample(wav, orig_sr = sr, target_sr = 44100), 44100, format = 'MP3')
    b.seek(0)
    res = soundfile.read(b)
    res = (librosa.resample(res[0], orig_sr = res[1], target_sr = sr).astype(wav.dtype), sr)
    if res[0].shape == (wav.shape[0] + 1,):
        res = (res[0][:-1], res[1])

    assert res[0].dtype == wav.dtype, f'{wav.dtype} -> {res[0].dtype}'
    assert res[0].shape == wav.shape, f'{wav.shape} -> {res[0].shape}'
    assert res[1] == sr, f'{sr} -> {res[1]}'
    return res[0]
