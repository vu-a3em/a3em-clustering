import soundfile
import pydub
import librosa
import numpy as np
import os
import uuid

def mp3_roundtrip(wav: np.ndarray, sr: int) -> np.ndarray:
    f = f'wmw-cvt-{uuid.uuid4()}'

    try:
        # doing the exact same thing in-memory gives garbled audio... whatever...
        soundfile.write(f, wav, sr, format = 'WAV')
        seg = pydub.AudioSegment.from_wav(f)
        seg.export(f, format = 'mp3')
        seg = pydub.AudioSegment.from_mp3(f)
        seg.export(f, format = 'wav')
        res = soundfile.read(f)
        res = (librosa.resample(res[0], orig_sr = res[1], target_sr = sr), sr)

        assert res[0].dtype == wav.dtype, f'{wav.dtype} -> {res[0].dtype}'
        assert res[0].shape == wav.shape, f'{wav.shape} -> {res[0].shape}'
        assert res[1] == sr, f'{sr} -> {res[1]}'
        return res[0]
    finally:
        try:
            os.remove(f)
        except:
            pass
