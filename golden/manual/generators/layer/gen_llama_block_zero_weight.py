#!/usr/bin/env python3
"""Generate golden data for the zero-weight LLaMA block layer smoke."""

from __future__ import annotations

import json
import struct
from pathlib import Path


CASES = [
    {
        "case_id": "llama_block_zero_weight_seq128_embed64",
        "seq_len": 128,
        "embed_dim": 64,
        "key_dim": 64,
        "value_dim": 64,
        "n_head_q": 1,
        "n_head_kv": 1,
        "ffn_dim": 64,
        "max_ctx_len": 128,
    },
    {
        "case_id": "llama_block_zero_weight_1b_shape_seq128",
        "seq_len": 128,
        "embed_dim": 2048,
        "key_dim": 64,
        "value_dim": 64,
        "n_head_q": 32,
        "n_head_kv": 8,
        "ffn_dim": 8192,
        "max_ctx_len": 8192,
    },
]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def input_value(row: int, col: int) -> float:
    v = ((row * 17 + col * 5) % 97) + 1
    return f32(v * 0.0009765625)


def v_weight_value(row: int, col: int) -> int:
    return ((row * 3 + col * 5) % 7) - 3


def causal_mask(seq_len: int, max_ctx_len: int) -> list[int]:
    stride = (max_ctx_len + 7) // 8
    mask = [0] * (seq_len * stride)
    for row in range(seq_len):
        for col in range(row + 1):
            bit = row * stride * 8 + col
            mask[bit // 8] |= 1 << (bit % 8)
    return [v - 256 if v >= 128 else v for v in mask]


def tensor_desc(path: str, dtype: str, element_bits: int,
                shape: list[int], stride_bytes: int) -> dict:
    total = element_bits // 8
    for dim in shape:
        total *= dim
    return {
        "path": path,
        "dtype": dtype,
        "element_bits": element_bits,
        "total_bytes": total,
        "layout": "row_major",
        "shape": shape,
        "stride_bytes": stride_bytes,
    }


def c_float(value: float) -> str:
    return f"{f32(value).hex()}f"


def c_items(values, line_items: int = 8) -> str:
    lines = []
    for i in range(0, len(values), line_items):
        chunk = values[i:i + line_items]
        lines.append("    " + ", ".join(str(v) for v in chunk) + ",")
    return "\n".join(lines)


def c_float_items(values, line_items: int = 4) -> str:
    lines = []
    for i in range(0, len(values), line_items):
        chunk = values[i:i + line_items]
        lines.append("    " + ", ".join(c_float(v) for v in chunk) + ",")
    return "\n".join(lines)


def emit_header(case: dict, out_root: Path) -> None:
    seq_len = case["seq_len"]
    embed_dim = case["embed_dim"]
    key_dim = case["key_dim"]
    value_dim = case["value_dim"]
    n_head_kv = case["n_head_kv"]
    max_ctx_len = case["max_ctx_len"]
    v_rows = n_head_kv * value_dim
    mask = causal_mask(seq_len, max_ctx_len)

    input_values = [
        input_value(row, col)
        for row in range(seq_len)
        for col in range(embed_dim)
    ]
    v_weight_values = [
        v_weight_value(row, col)
        for row in range(v_rows)
        for col in range(embed_dim)
    ]

    guard = f"GOLDEN_{case['case_id'].upper()}_INPUTS_H"
    text = f"""#ifndef {guard}
#define {guard}

#include <stdint.h>

#define GOLDEN_LLAMA_SEQ_LEN {seq_len}
#define GOLDEN_LLAMA_EMBED_DIM {embed_dim}
#define GOLDEN_LLAMA_KEY_DIM {key_dim}
#define GOLDEN_LLAMA_VALUE_DIM {value_dim}
#define GOLDEN_LLAMA_N_HEAD_Q {case["n_head_q"]}
#define GOLDEN_LLAMA_N_HEAD_KV {case["n_head_kv"]}
#define GOLDEN_LLAMA_FFN_DIM {case["ffn_dim"]}
#define GOLDEN_LLAMA_MAX_CTX_LEN {max_ctx_len}
#define GOLDEN_LLAMA_CAUSAL_MASK_BYTES {len(mask)}

static const float golden_llama_input[{seq_len * embed_dim}]
    __attribute__((aligned(64))) = {{
{c_float_items(input_values)}
}};

static const float golden_llama_attn_norm_weight[{embed_dim}]
    __attribute__((aligned(64))) = {{
{c_float_items([1.0] * embed_dim)}
}};

static const float golden_llama_ffn_norm_weight[{embed_dim}]
    __attribute__((aligned(64))) = {{
{c_float_items([1.0] * embed_dim)}
}};

static const int8_t golden_llama_proj_v_weight[{v_rows * embed_dim}]
    __attribute__((aligned(64))) = {{
{c_items(v_weight_values, 16)}
}};

static const float golden_llama_proj_q_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};
static const float golden_llama_proj_k_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};
static const float golden_llama_proj_v_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};
static const float golden_llama_proj_o_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};
static const float golden_llama_ffn_gate_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};
static const float golden_llama_ffn_up_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};
static const float golden_llama_ffn_down_scale[1] __attribute__((aligned(64))) = {{
    {c_float(0.001)},
}};

static const float golden_llama_rope_theta[{key_dim // 2}]
    __attribute__((aligned(64))) = {{
{c_float_items([1.0] * (key_dim // 2))}
}};

static const int8_t golden_llama_causal_mask[{len(mask)}]
    __attribute__((aligned(64))) = {{
{c_items(mask, 16)}
}};

#endif /* {guard} */
"""
    (out_root / "golden_llama_block_inputs.h").write_text(text)


def emit_case(case: dict) -> None:
    case_id = case["case_id"]
    seq_len = case["seq_len"]
    embed_dim = case["embed_dim"]
    out_root = Path("golden/manual/layer") / case_id
    out_root.mkdir(parents=True, exist_ok=True)
    output = [
        input_value(row, col)
        for row in range(seq_len)
        for col in range(embed_dim)
    ]
    (out_root / "golden_output.bin").write_bytes(
        struct.pack(f"<{len(output)}f", *output)
    )
    emit_header(case, out_root)
    manifest = {
        "id": case_id,
        "op": "llama_block",
        "level": "layer",
        "tensors": {
            "golden_output": tensor_desc(
                "golden_output.bin",
                "F32",
                32,
                [seq_len, embed_dim],
                embed_dim * 4,
            )
        },
        "attributes": {
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "key_dim": case["key_dim"],
            "value_dim": case["value_dim"],
            "n_head_q": case["n_head_q"],
            "n_head_kv": case["n_head_kv"],
            "ffn_dim": case["ffn_dim"],
            "input": "deterministic nonzero generated in golden_llama_block_inputs.h",
            "weights": "Q/K/O and FFN weights are zero; V uses deterministic nonzero weights",
            "expected": "two residual adds preserve input",
        },
        "generator": {
            "tool": "golden/manual/generators/layer/gen_llama_block_zero_weight.py",
            "created": "2026-05-19",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    for case in CASES:
        emit_case(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
