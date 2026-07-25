#!/usr/bin/env python3
"""Derive parity_short and parity_long_quiet from parity_jfk.wav (stdlib only)."""
import wave, numpy as np, pathlib
d = pathlib.Path(__file__).parent
def rd(p):
    w = wave.open(str(p))
    return np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
def wr(p, x, sr=16000):
    w = wave.open(str(p), 'w'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes()); w.close()
jfk, sr = rd(d / 'parity_jfk.wav'), 16000
wr(d / 'parity_short.wav', jfk[:12000])                       # 0.75 s
long_ = np.tile(jfk, int(np.ceil(120 * sr / len(jfk))))[:130 * sr]
wr(d / 'parity_long_quiet.wav', long_ * (10 ** (-30 / 20) / np.sqrt((long_ ** 2).mean())))
print('wrote parity_short.wav, parity_long_quiet.wav')
