#!/usr/bin/env python3
"""Dump nemo128.onnx reference mel tensors for the Rust parity test."""
import wave, json, hashlib, pathlib, numpy as np, onnxruntime as ort
d = pathlib.Path(__file__).parent
out = pathlib.Path.home() / 'models' / 'parakeet' / 'ref'; out.mkdir(parents=True, exist_ok=True)
sess = ort.InferenceSession(str(pathlib.Path.home() / 'models' / 'parakeet' / 'nemo128.onnx'),
                            providers=['CPUExecutionProvider'])
meta = {}
for name in ['parity_jfk', 'parity_short', 'parity_long_quiet']:
    p = d / f'{name}.wav'; w = wave.open(str(p))
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
    feats, _ = sess.run(None, {'waveforms': x[None, :].astype(np.float32),
                               'waveforms_lens': np.array([len(x)], dtype=np.int64)})
    mel = feats[0]
    x.astype('<f4').tofile(out / f'{name}_audio.f32')
    mel.astype('<f4').tofile(out / f'{name}_mel.f32')
    meta[name] = {'samples': int(len(x)), 'mel_bins': int(mel.shape[0]), 'frames': int(mel.shape[1]),
                  'wav_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    print(f'{name}: mel {mel.shape}')
json.dump(meta, open(out / 'manifest.json', 'w'), indent=2)
