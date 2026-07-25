#!/usr/bin/env python3
"""Dump the reference TDT greedy decode for the Rust decoder parity test.

Drives `decoder_joint-model.onnx` step by step over the reference encoder
output and records **every** intermediate the Rust side can diverge at:

  decode/joint_logits.f32   [steps, 8198]  the joint output at each loop step
  decode/pred_out.f32       [steps, 640]   prednet output feeding each step
  decode/states_h.f32       [steps, 2, 640] LSTM h AFTER each step (pre-commit)
  decode/states_c.f32       [steps, 2, 640] LSTM c AFTER each step
  decode/manifest.json      tokens, timestamps, durations, per-step trace

Dumping every step is the same trick that made the encoder stage cheap: a
token mismatch in Rust becomes a direct lookup of the first step whose joint
logits diverge, instead of a bisection over the decode.

The loop is `onnx_asr._AsrWithTransducerDecoding._decoding` — the decoder that
ships with these exact exports, so it is the oracle for loop semantics as well
as for tensors.
"""

import json
import pathlib

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path.home() / "models" / "parakeet"
REF = ROOT / "ref"
OUT = REF / "decode"
OUT.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 8193  # 8192 SPE tokens + <blk>
BLANK = 8192
DURATIONS = [0, 1, 2, 3, 4]
MAX_TOKENS_PER_STEP = 10
D_MODEL = 1024
PRED_HIDDEN = 640

sess = ort.InferenceSession(str(ROOT / "decoder_joint-model.onnx"), providers=["CPUExecutionProvider"])

# The instrumented encoder dump is [1, 1024, T] (channel-major, as the graph
# emits it). Keep it in that layout — it is what decoder_joint wants per frame.
enc = np.fromfile(REF / "enc" / "outputs.f32", dtype="<f4")
frames = enc.size // D_MODEL
enc = enc.reshape(D_MODEL, frames)
print(f"encoder output: {D_MODEL} x {frames}")

# Expose every sublayer boundary, so a Rust divergence names the guilty stage
# instead of leaving "prednet or joint?" open. Ordered along the data path.
import onnx  # noqa: E402

STAGES = {
    "embed": "/decoder/embed/Gather_output_0",
    "lstm0": "/decoder/dec_rnn/lstm/Squeeze_output_0",
    "lstm1": "/decoder/dec_rnn/lstm/Squeeze_1_output_0",
    "enc_proj": "/joint/enc/Add_output_0",
    "pred_proj": "/joint/pred/Add_output_0",
    "joint_sum": "/joint/Add_output_0",
    "joint_relu": "/joint/joint_net/joint_net.0/Relu_output_0",
}

graph = onnx.load(str(ROOT / "decoder_joint-model.onnx"), load_external_data=False)
produced = {o for n in graph.graph.node for o in n.output}
existing = {o.name for o in graph.graph.output}
for label, name in STAGES.items():
    if name not in produced:
        raise RuntimeError(f"stage {label}: {name} is not produced by any node")
    if name not in existing:
        graph.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
inst_path = ROOT / "decoder_joint-instrumented.onnx"
onnx.save(graph, str(inst_path))
inst = ort.InferenceSession(str(inst_path), providers=["CPUExecutionProvider"])
inst_names = [o.name for o in inst.get_outputs()]
print("instrumented outputs:", inst_names)
stage_rows: dict[str, list] = {k: [] for k in STAGES}

state1 = np.zeros((2, 1, PRED_HIDDEN), dtype=np.float32)
state2 = np.zeros((2, 1, PRED_HIDDEN), dtype=np.float32)

tokens: list[int] = []
timestamps: list[int] = []
trace = []
joint_rows = []
pred_rows = []
h_rows = []
c_rows = []

t = 0
emitted = 0
steps = 0
while t < frames:
    prev = tokens[-1] if tokens else BLANK
    feed = {
        "encoder_outputs": enc[None, :, t : t + 1],
        "targets": np.array([[prev]], dtype=np.int32),
        "target_length": np.array([1], dtype=np.int32),
        "input_states_1": state1,
        "input_states_2": state2,
    }
    res = dict(zip(inst_names, inst.run(inst_names, feed)))
    out = np.squeeze(res["outputs"])
    s1, s2 = res["output_states_1"], res["output_states_2"]

    logits = out[:VOCAB_SIZE]
    dur_logits = out[VOCAB_SIZE:]
    token = int(logits.argmax())
    d_idx = int(dur_logits.argmax())
    step = DURATIONS[d_idx]

    # Duration-argmax margin: the number that decides whether token-exactness
    # is achievable at all (recon's single biggest unknown).
    srt = np.sort(dur_logits)[::-1]
    margin = float(srt[0] - srt[1])

    joint_rows.append(out.astype("<f4"))
    for label, name in STAGES.items():
        stage_rows[label].append(np.squeeze(res[name]).astype("<f4"))
    pred_rows.append(np.squeeze(res[STAGES["lstm1"]]).astype("<f4"))
    h_rows.append(s1[:, 0, :].astype("<f4"))
    c_rows.append(s2[:, 0, :].astype("<f4"))

    trace.append(
        {
            "step": steps,
            "t": t,
            "prev_token": prev,
            "token": token,
            "duration_index": d_idx,
            "duration": step,
            "duration_margin": margin,
            "emitted_before": emitted,
        }
    )

    if token != BLANK:
        state1, state2 = s1, s2  # commit ONLY on non-blank
        tokens.append(token)
        timestamps.append(t)
        emitted += 1

    if step > 0:
        t += step
        emitted = 0
    elif token == BLANK or emitted == MAX_TOKENS_PER_STEP:
        t += 1
        emitted = 0
    steps += 1
    if steps > 20000:
        raise RuntimeError("decode did not terminate")

vocab = {}
for line in (ROOT / "vocab.txt").read_text(encoding="utf-8").splitlines():
    piece, idx = line.rsplit(" ", 1)
    vocab[int(idx)] = piece.replace("▁", " ")
text = "".join(vocab[i] for i in tokens).strip()

np.stack(joint_rows).tofile(OUT / "joint_logits.f32")
np.stack(pred_rows).tofile(OUT / "pred_out.f32")
np.stack(h_rows).tofile(OUT / "states_h.f32")
np.stack(c_rows).tofile(OUT / "states_c.f32")
stage_shapes = {}
for label, rows in stage_rows.items():
    arr = np.stack(rows)
    arr.tofile(OUT / f"stage_{label}.f32")
    stage_shapes[label] = list(arr.shape)

margins = [x["duration_margin"] for x in trace]
manifest = {
    "fixture": "parity_jfk",
    "encoder_frames": frames,
    "steps": steps,
    "tokens": tokens,
    "timestamps": timestamps,
    "durations": [x["duration"] for x in trace],
    "blank": BLANK,
    "vocab_size": VOCAB_SIZE,
    "durations_table": DURATIONS,
    "max_tokens_per_step": MAX_TOKENS_PER_STEP,
    "min_duration_margin": min(margins),
    "text": text,
    "trace": trace,
    "joint_logits_shape": [steps, 8198],
    "pred_out_shape": [len(pred_rows), PRED_HIDDEN],
    "states_shape": [steps, 2, PRED_HIDDEN],
    "stage_shapes": stage_shapes,
}
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)

# Flat sidecars so the Rust test needs no JSON parser. The golden decode is
# three integer sequences and a string; keeping them as raw i32/UTF-8 keeps the
# test's failure messages about the decode rather than about deserialization.
np.array(tokens, dtype="<i4").tofile(OUT / "tokens.i32")
np.array(timestamps, dtype="<i4").tofile(OUT / "frames.i32")
np.array([x["duration"] for x in trace], dtype="<i4").tofile(OUT / "durations.i32")
np.array([x["t"] for x in trace], dtype="<i4").tofile(OUT / "step_frames.i32")
np.array([x["token"] for x in trace], dtype="<i4").tofile(OUT / "step_tokens.i32")
(OUT / "text.txt").write_text(text, encoding="utf-8")

print(f"steps={steps} tokens={len(tokens)} min_duration_margin={min(margins):.4f}")
print(f"text: {text!r}")
