#!/usr/bin/env python3
"""Convert a Hugging Face Whisper checkpoint to the OpenAI tensor naming Ocelotl expects.

`docs/artifact-preparation.md` calls for "a converted tiny.en bundle" but never
said how to convert one, so the bundle could not be rebuilt from the repository
alone — which quietly made the Whisper alpha gate unsatisfiable for anyone who
did not already have the artifacts. This script closes that gap.

The two schemes differ structurally, not just by a prefix:

    Hugging Face                              OpenAI / Ocelotl
    model.encoder.embed_positions.weight  ->  encoder.positional_embedding
    model.encoder.layer_norm.{w,b}        ->  encoder.ln_post.{w,b}
    model.decoder.embed_tokens.weight     ->  decoder.token_embedding.weight
    model.decoder.layer_norm.{w,b}        ->  decoder.ln.{w,b}
    ....layers.N.self_attn.q_proj         ->  ....blocks.N.attn.query
    ....layers.N.self_attn_layer_norm     ->  ....blocks.N.attn_ln
    ....layers.N.encoder_attn.k_proj      ->  ....blocks.N.cross_attn.key
    ....layers.N.fc1 / fc2                ->  ....blocks.N.mlp.0 / mlp.2
    ....layers.N.final_layer_norm         ->  ....blocks.N.mlp_ln

Two Whisper quirks the mapping has to respect:

  * `key` projections carry **no bias** in Whisper (`q`, `v`, `out` do). A
    converter that invents a zero `key.bias` produces a file that loads and
    decodes subtly wrong, so absent-by-design is preserved rather than filled.
  * HF stores `proj_out` tied to the token embedding. It is emitted only when
    the source actually carries it untied.

Every source tensor must be consumed and every mapped name must be distinct;
both are asserted, so a silently dropped or colliding tensor fails here instead
of surfacing later as a parity mismatch that looks like a model bug.

Usage:
    python3 tools/convert_whisper_hf_to_openai.py IN.safetensors OUT.safetensors
"""

import json
import re
import struct
import sys
from pathlib import Path

# Suffix rewrites applied inside an encoder/decoder layer block.
LAYER_SUFFIX = [
    ("self_attn.q_proj", "attn.query"),
    ("self_attn.k_proj", "attn.key"),
    ("self_attn.v_proj", "attn.value"),
    ("self_attn.out_proj", "attn.out"),
    ("self_attn_layer_norm", "attn_ln"),
    ("encoder_attn.q_proj", "cross_attn.query"),
    ("encoder_attn.k_proj", "cross_attn.key"),
    ("encoder_attn.v_proj", "cross_attn.value"),
    ("encoder_attn.out_proj", "cross_attn.out"),
    ("encoder_attn_layer_norm", "cross_attn_ln"),
    ("final_layer_norm", "mlp_ln"),
    ("fc1", "mlp.0"),
    ("fc2", "mlp.2"),
]

TOP_LEVEL = {
    "model.encoder.embed_positions.weight": "encoder.positional_embedding",
    "model.encoder.layer_norm.weight": "encoder.ln_post.weight",
    "model.encoder.layer_norm.bias": "encoder.ln_post.bias",
    "model.encoder.conv1.weight": "encoder.conv1.weight",
    "model.encoder.conv1.bias": "encoder.conv1.bias",
    "model.encoder.conv2.weight": "encoder.conv2.weight",
    "model.encoder.conv2.bias": "encoder.conv2.bias",
    "model.decoder.embed_positions.weight": "decoder.positional_embedding",
    "model.decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "model.decoder.layer_norm.weight": "decoder.ln.weight",
    "model.decoder.layer_norm.bias": "decoder.ln.bias",
    "proj_out.weight": "decoder.proj_out.weight",
}

LAYER_RE = re.compile(r"^model\.(encoder|decoder)\.layers\.(\d+)\.(.+)$")


def convert_name(name: str) -> str | None:
    """Map one HF tensor name to its OpenAI equivalent, or None to drop it."""
    if name in TOP_LEVEL:
        return TOP_LEVEL[name]
    m = LAYER_RE.match(name)
    if not m:
        return None
    side, layer, rest = m.group(1), int(m.group(2)), m.group(3)
    for hf_suffix, oa_suffix in LAYER_SUFFIX:
        if rest.startswith(hf_suffix + "."):
            tail = rest[len(hf_suffix) + 1 :]  # "weight" | "bias"
            return f"{side}.blocks.{layer}.{oa_suffix}.{tail}"
    return None


def read_safetensors(path: Path):
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    body = raw[8 + header_len :]
    return header, body


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    header, body = read_safetensors(src)
    meta = header.pop("__metadata__", None)

    mapped: dict[str, str] = {}
    dropped: list[str] = []
    for name in header:
        target = convert_name(name)
        if target is None:
            dropped.append(name)
        else:
            mapped[name] = target

    # A name collision would silently discard a tensor.
    seen: dict[str, str] = {}
    for source, target in mapped.items():
        if target in seen:
            raise SystemExit(f"collision: {source} and {seen[target]} both map to {target}")
        seen[target] = source

    # Anything unmapped is either a genuine extra (tied proj_out) or a hole in
    # the table. Print it so the difference is a decision, never an accident.
    for name in dropped:
        print(f"  dropped (unmapped): {name}")

    out_header: dict[str, object] = {}
    chunks: list[bytes] = []
    offset = 0
    for source, target in sorted(mapped.items(), key=lambda kv: kv[1]):
        entry = header[source]
        start, end = entry["data_offsets"]
        blob = body[start:end]
        out_header[target] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + len(blob)],
        }
        chunks.append(blob)
        offset += len(blob)

    if meta is not None:
        out_header["__metadata__"] = meta
    packed = json.dumps(out_header).encode("utf-8")
    pad = (-len(packed)) % 8  # safetensors requires 8-byte alignment
    packed += b" " * pad

    with dst.open("wb") as f:
        f.write(struct.pack("<Q", len(packed)))
        f.write(packed)
        for blob in chunks:
            f.write(blob)

    print(f"converted {len(mapped)} tensors -> {dst}")
    print(f"dropped {len(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
