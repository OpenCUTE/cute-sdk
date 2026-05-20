#!/usr/bin/env python3
"""Generate golden data for a full-shape nonzero-weight LLaMA block case.

Large tensors are emitted as raw .bin files. The test binary embeds them with
.incbin so verilator does not spend time initializing 1B-shape weights.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np


CASE_ID = "llama_block_nonzero_weight_1b_shape_seq128"
OUT_ROOT = Path("golden/manual/layer") / CASE_ID

SEQ_LEN = 128
EMBED_DIM = 2048
KEY_DIM = 64
VALUE_DIM = 64
N_HEAD_Q = 32
N_HEAD_KV = 8
FFN_DIM = 8192
MAX_CTX_LEN = 8192
RMS_EPSILON = np.float32(1.0e-5)
KV_SCALE = np.float32(0.125)
GROUP = 64

PROJ_Q_SCALE = np.float32(5.0e-5)
PROJ_K_SCALE = np.float32(5.0e-5)
PROJ_V_SCALE = np.float32(5.0e-5)
PROJ_O_SCALE = np.float32(5.0e-4)
FFN_GATE_SCALE = np.float32(5.0e-5)
FFN_UP_SCALE = np.float32(5.0e-5)
FFN_DOWN_SCALE = np.float32(5.0e-4)

WEIGHTS = {
    "proj_q": (N_HEAD_Q * KEY_DIM, EMBED_DIM, 3),
    "proj_k": (N_HEAD_KV * KEY_DIM, EMBED_DIM, 5),
    "proj_v": (N_HEAD_KV * VALUE_DIM, EMBED_DIM, 7),
    "proj_o": (EMBED_DIM, EMBED_DIM, 11),
    "ffn_gate": (FFN_DIM, EMBED_DIM, 13),
    "ffn_up": (FFN_DIM, EMBED_DIM, 17),
    "ffn_down": (EMBED_DIM, FFN_DIM, 19),
}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32a(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def to_bf16_f32(value: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    bits = arr.view(np.uint32)
    truncated = (bits & np.uint32(0xFFFF0000)).astype(np.uint32, copy=False)
    return truncated.view(np.float32).reshape(arr.shape)


def c_float(value: float) -> str:
    return f"{f32(value).hex()}f"


def c_items(values, line_items: int = 8) -> str:
    lines = []
    for i in range(0, len(values), line_items):
        chunk = values[i:i + line_items]
        lines.append("    " + ", ".join(str(int(v)) for v in chunk) + ",")
    return "\n".join(lines)


def c_float_items(values, line_items: int = 4) -> str:
    lines = []
    flat = list(values)
    for i in range(0, len(flat), line_items):
        chunk = flat[i:i + line_items]
        lines.append("    " + ", ".join(c_float(float(v)) for v in chunk) + ",")
    return "\n".join(lines)


def input_matrix() -> np.ndarray:
    rows = np.arange(SEQ_LEN, dtype=np.int32)[:, None]
    cols = np.arange(EMBED_DIM, dtype=np.int32)[None, :]
    values = (((rows * 17 + cols * 5) % 97) + 1).astype(np.float32)
    return f32a(values * np.float32(0.0009765625))


def norm_weight(seed: int) -> np.ndarray:
    cols = np.arange(EMBED_DIM, dtype=np.int32)
    values = np.float32(0.75) + (((cols * 3 + seed) % 17).astype(np.float32)
                                 * np.float32(0.01))
    return f32a(values)


def rope_theta() -> np.ndarray:
    half = KEY_DIM // 2
    values = [
        f32(math.pow(10000.0, -2.0 * float(k) / float(KEY_DIM)))
        for k in range(half)
    ]
    return np.asarray(values, dtype=np.float32)


def causal_mask() -> list[int]:
    stride = (MAX_CTX_LEN + 7) // 8
    mask = [0] * (SEQ_LEN * stride)
    for row in range(SEQ_LEN):
        for col in range(row + 1):
            bit = row * stride * 8 + col
            mask[bit // 8] |= 1 << (bit % 8)
    return [v - 256 if v >= 128 else v for v in mask]


def weight_table(seed: int) -> np.ndarray:
    rows = np.arange(GROUP, dtype=np.int32)[:, None]
    cols = np.arange(GROUP, dtype=np.int32)[None, :]
    values = ((rows * 17 + cols * 31 + seed) % 9) - 4
    values = np.where(values == 0, 1, values)
    return values.astype(np.int8)


def emit_weight_bin(name: str, out_dim: int, in_dim: int, seed: int) -> None:
    table = weight_table(seed)
    row_mod = np.arange(out_dim, dtype=np.int32) % GROUP
    col_mod = np.arange(in_dim, dtype=np.int32) % GROUP
    weights = table[row_mod[:, None], col_mod[None, :]].astype(np.int8)
    (OUT_ROOT / f"{name}_weight.bin").write_bytes(weights.tobytes())


def grouped_matmul_i8(a: np.ndarray, out_dim: int, seed: int) -> np.ndarray:
    assert a.shape[1] % GROUP == 0
    table = weight_table(seed).astype(np.int32)
    grouped = a.astype(np.int32).reshape(a.shape[0], -1, GROUP).sum(axis=1)
    unique = grouped @ table.T
    return unique[:, np.arange(out_dim, dtype=np.int32) % GROUP].astype(np.int32)


def dequant(acc: np.ndarray, row_scale: np.ndarray, weight_scale: np.float32) -> np.ndarray:
    return f32a(acc.astype(np.float32) * row_scale[:, None] * weight_scale)


def rmsnorm_with_scale(input_data: np.ndarray,
                       weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sq = f32a(input_data * input_data)
    mean = f32a(np.sum(sq, axis=1, dtype=np.float32) / np.float32(input_data.shape[1]))
    rms = f32a(np.float32(1.0) / np.sqrt(f32a(mean + RMS_EPSILON), dtype=np.float32))
    output = f32a(f32a(input_data * rms[:, None]) * weight[None, :])
    scale = f32a(np.max(np.abs(output), axis=1) / np.float32(127.0))
    return output, scale


def smoothquant_stage2(input_data: np.ndarray, scale: np.ndarray) -> np.ndarray:
    output = np.zeros(input_data.shape, dtype=np.int8)
    valid = scale != np.float32(0.0)
    if np.any(valid):
        q = np.rint(input_data[valid] / scale[valid, None])
        q = np.clip(q, -128, 127)
        output[valid] = q.astype(np.int8)
    return output


def smoothquant(input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = f32a(np.max(np.abs(input_data), axis=1) / np.float32(127.0))
    return smoothquant_stage2(input_data, scale), scale


def bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return bits.astype(np.uint32).view(np.float32)


def round_magic(value: np.ndarray) -> np.ndarray:
    magic = np.float32(float.fromhex("0x1.8p23"))
    return f32a(f32a(value + magic) - magic)


def exp_approx(x: np.ndarray) -> np.ndarray:
    neg_ln2 = np.float32(-0.69314718056)
    inv_ln2 = np.float32(1.44269504089)
    af = f32a(x * inv_ln2)
    a = round_magic(af)
    a_int = a.astype(np.int32)
    mask_max = a_int > 127
    mask_min = a_int < -126
    exponent = ((a_int + 127).astype(np.uint32) << np.uint32(23))
    a2 = bits_to_f32(exponent)
    b = f32a(x + f32a(neg_ln2 * a))

    c0 = np.float32(1.0)
    c1 = np.float32(1.0)
    c2 = np.float32(0.5)
    c3 = np.float32(0.166666666667)
    c4 = np.float32(0.041666666667)
    c5 = np.float32(0.008333333333)
    c6 = np.float32(0.001388888889)
    p = f32a(c5 + f32a(c6 * b))
    p = f32a(c4 + f32a(p * b))
    p = f32a(c3 + f32a(p * b))
    p = f32a(c2 + f32a(p * b))
    p = f32a(c1 + f32a(p * b))
    p = f32a(c0 + f32a(p * b))
    p = f32a(a2 * p)
    p = np.where(mask_max, np.float32(np.inf), p)
    p = np.where(mask_min, np.float32(0.0), p)
    return f32a(p)


def sin_small_approx(x: np.ndarray) -> np.ndarray:
    x2 = f32a(x * x)
    c1 = np.float32(1.0)
    c3 = np.float32(-0.166666666667)
    c5 = np.float32(0.008333333333)
    c7 = np.float32(-0.0001984126984)
    c9 = np.float32(0.000002755731922)
    c11 = np.float32(-0.000000025052108)
    result = f32a(c9 + f32a(c11 * x2))
    result = f32a(c7 + f32a(result * x2))
    result = f32a(c5 + f32a(result * x2))
    result = f32a(c3 + f32a(result * x2))
    result = f32a(c1 + f32a(result * x2))
    return f32a(result * x)


def sin_approx(x: np.ndarray) -> np.ndarray:
    pi = np.float32(3.14159265359)
    pi_div_2 = np.float32(pi / np.float32(2.0))
    inv_pi = np.float32(0.31830988618)
    new_rad = f32a(x + pi_div_2)
    round_v = round_magic(f32a(new_rad * inv_pi))
    new_rad = f32a(new_rad - f32a(round_v * pi))
    new_rad = f32a(new_rad - pi_div_2)
    result = sin_small_approx(new_rad)
    round_int = round_v.astype(np.uint32)
    return f32a(np.where((round_int & np.uint32(1)) != 0, -result, result))


def cos_approx(x: np.ndarray) -> np.ndarray:
    pi = np.float32(3.14159265359)
    pi_div_2 = np.float32(pi / np.float32(2.0))
    return sin_approx(f32a(x + pi_div_2))


def silu(input_data: np.ndarray) -> np.ndarray:
    denom = f32a(np.float32(1.0) + exp_approx(f32a(-input_data)))
    return f32a(input_data / denom)


def apply_rope(dequant_data: np.ndarray, total_dim: int) -> np.ndarray:
    output = np.zeros(dequant_data.shape, dtype=np.float32)
    theta = rope_theta()
    positions = np.arange(SEQ_LEN, dtype=np.float32)[:, None]
    angle = f32a(positions * theta[None, :])
    sin_v = sin_approx(angle)
    cos_v = cos_approx(angle)
    for head0 in range(0, total_dim, KEY_DIM):
        chunk = dequant_data[:, head0:head0 + KEY_DIM]
        real_in = chunk[:, 0::2]
        imag_in = chunk[:, 1::2]
        output[:, head0:head0 + KEY_DIM // 2] = f32a(
            f32a(real_in * cos_v) - f32a(imag_in * sin_v)
        )
        output[:, head0 + KEY_DIM // 2:head0 + KEY_DIM] = f32a(
            f32a(real_in * sin_v) + f32a(imag_in * cos_v)
        )
    return to_bf16_f32(output)


def masked_softmax_bf16(scores: np.ndarray) -> np.ndarray:
    scaled = f32a(scores * KV_SCALE)
    col = np.arange(SEQ_LEN, dtype=np.int32)[None, :]
    row = np.arange(SEQ_LEN, dtype=np.int32)[:, None]
    scaled = np.where(col <= row, scaled, np.float32(-np.inf)).astype(np.float32)
    max_val = np.max(scaled, axis=1).astype(np.float32)
    shifted = f32a(scaled - max_val[:, None])
    shifted = np.where(np.isneginf(scaled), np.float32(-90.0), shifted)
    exps = exp_approx(shifted)
    denom = f32a(np.sum(exps, axis=1, dtype=np.float32))
    return to_bf16_f32(f32a(exps / denom[:, None]))


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


def emit_i8_bin(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.asarray(array, dtype=np.int8).tobytes())


def emit_bf16_bin(path: Path, array: np.ndarray) -> None:
    arr = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    bf16 = (arr.view(np.uint32) >> np.uint32(16)).astype("<u2", copy=False)
    path.write_bytes(bf16.tobytes())


def emit_f32_bin(path: Path, array: np.ndarray) -> None:
    path.write_bytes(np.asarray(array, dtype="<f4").tobytes())


def emit_header(attn_weight: np.ndarray, ffn_weight: np.ndarray) -> None:
    mask = causal_mask()
    guard = "GOLDEN_LLAMA_BLOCK_NONZERO_WEIGHT_1B_SHAPE_SEQ128_INPUTS_H"
    text = f"""#ifndef {guard}
#define {guard}

#include <stdint.h>

#define GOLDEN_LLAMA_SEQ_LEN {SEQ_LEN}
#define GOLDEN_LLAMA_EMBED_DIM {EMBED_DIM}
#define GOLDEN_LLAMA_KEY_DIM {KEY_DIM}
#define GOLDEN_LLAMA_VALUE_DIM {VALUE_DIM}
#define GOLDEN_LLAMA_N_HEAD_Q {N_HEAD_Q}
#define GOLDEN_LLAMA_N_HEAD_KV {N_HEAD_KV}
#define GOLDEN_LLAMA_FFN_DIM {FFN_DIM}
#define GOLDEN_LLAMA_MAX_CTX_LEN {MAX_CTX_LEN}
#define GOLDEN_LLAMA_CAUSAL_MASK_BYTES {len(mask)}

extern const float golden_llama_input[{SEQ_LEN * EMBED_DIM}];
extern const int8_t golden_llama_proj_q_weight[{N_HEAD_Q * KEY_DIM * EMBED_DIM}];
extern const int8_t golden_llama_proj_k_weight[{N_HEAD_KV * KEY_DIM * EMBED_DIM}];
extern const int8_t golden_llama_proj_v_weight[{N_HEAD_KV * VALUE_DIM * EMBED_DIM}];
extern const int8_t golden_llama_proj_o_weight[{EMBED_DIM * EMBED_DIM}];
extern const int8_t golden_llama_ffn_gate_weight[{FFN_DIM * EMBED_DIM}];
extern const int8_t golden_llama_ffn_up_weight[{FFN_DIM * EMBED_DIM}];
extern const int8_t golden_llama_ffn_down_weight[{EMBED_DIM * FFN_DIM}];
extern const int8_t golden_llama_attn_norm_q8[{SEQ_LEN * EMBED_DIM}];
extern const float golden_llama_attn_norm_scale[{SEQ_LEN}];
extern const uint16_t golden_llama_q_bf16[{SEQ_LEN * N_HEAD_Q * KEY_DIM}];
extern const uint16_t golden_llama_k_bf16[{SEQ_LEN * N_HEAD_KV * KEY_DIM}];
extern const uint16_t golden_llama_v_bf16_t[{N_HEAD_KV * VALUE_DIM * SEQ_LEN}];
extern const uint16_t golden_llama_scores_head0_bf16[{SEQ_LEN * SEQ_LEN}];
extern const float golden_llama_proj_o_f32[{SEQ_LEN * EMBED_DIM}];
extern const int8_t golden_llama_ffn_norm_q8[{SEQ_LEN * EMBED_DIM}];
extern const float golden_llama_ffn_norm_scale[{SEQ_LEN}];
extern const float golden_llama_ffn_gate_f32[{SEQ_LEN * FFN_DIM}];
extern const float golden_llama_ffn_up_f32[{SEQ_LEN * FFN_DIM}];
extern const float golden_llama_ffn_up_row_scale[{SEQ_LEN}];
extern const int8_t golden_llama_ffn_up_q8[{SEQ_LEN * FFN_DIM}];

static const float golden_llama_attn_norm_weight[{EMBED_DIM}]
    __attribute__((aligned(64))) = {{
{c_float_items(attn_weight)}
}};

static const float golden_llama_ffn_norm_weight[{EMBED_DIM}]
    __attribute__((aligned(64))) = {{
{c_float_items(ffn_weight)}
}};

static const float golden_llama_proj_q_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(PROJ_Q_SCALE))},
}};
static const float golden_llama_proj_k_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(PROJ_K_SCALE))},
}};
static const float golden_llama_proj_v_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(PROJ_V_SCALE))},
}};
static const float golden_llama_proj_o_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(PROJ_O_SCALE))},
}};
static const float golden_llama_ffn_gate_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(FFN_GATE_SCALE))},
}};
static const float golden_llama_ffn_up_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(FFN_UP_SCALE))},
}};
static const float golden_llama_ffn_down_scale[1] __attribute__((aligned(64))) = {{
    {c_float(float(FFN_DOWN_SCALE))},
}};

static const float golden_llama_rope_theta[{KEY_DIM // 2}]
    __attribute__((aligned(64))) = {{
{c_float_items(rope_theta())}
}};

static const int8_t golden_llama_causal_mask[{len(mask)}]
    __attribute__((aligned(64))) = {{
{c_items(mask, 16)}
}};

#endif /* {guard} */
"""
    (OUT_ROOT / "golden_llama_block_inputs.h").write_text(text)


def emit_manifest() -> None:
    manifest = {
        "id": CASE_ID,
        "op": "llama_block",
        "level": "layer",
        "tensors": {
            "golden_output": tensor_desc(
                "golden_output.bin",
                "F32",
                32,
                [SEQ_LEN, EMBED_DIM],
                EMBED_DIM * 4,
            ),
            "golden_attn_norm_q8": tensor_desc(
                "attn_norm_q8.bin",
                "I8",
                8,
                [SEQ_LEN, EMBED_DIM],
                EMBED_DIM,
            ),
            "golden_attn_norm_scale": tensor_desc(
                "attn_norm_scale.bin",
                "F32",
                32,
                [SEQ_LEN],
                SEQ_LEN * 4,
            ),
            "golden_q_bf16": tensor_desc(
                "q_bf16.bin",
                "BF16",
                16,
                [SEQ_LEN, N_HEAD_Q * KEY_DIM],
                N_HEAD_Q * KEY_DIM * 2,
            ),
            "golden_k_bf16": tensor_desc(
                "k_bf16.bin",
                "BF16",
                16,
                [SEQ_LEN, N_HEAD_KV * KEY_DIM],
                N_HEAD_KV * KEY_DIM * 2,
            ),
            "golden_v_bf16_t": tensor_desc(
                "v_bf16_t.bin",
                "BF16",
                16,
                [N_HEAD_KV * VALUE_DIM, SEQ_LEN],
                SEQ_LEN * 2,
            ),
            "golden_scores_head0_bf16": tensor_desc(
                "scores_head0_bf16.bin",
                "BF16",
                16,
                [SEQ_LEN, SEQ_LEN],
                SEQ_LEN * 2,
            ),
            "golden_proj_o_f32": tensor_desc(
                "proj_o_f32.bin",
                "F32",
                32,
                [SEQ_LEN, EMBED_DIM],
                EMBED_DIM * 4,
            ),
            "golden_ffn_norm_q8": tensor_desc(
                "ffn_norm_q8.bin",
                "I8",
                8,
                [SEQ_LEN, EMBED_DIM],
                EMBED_DIM,
            ),
            "golden_ffn_norm_scale": tensor_desc(
                "ffn_norm_scale.bin",
                "F32",
                32,
                [SEQ_LEN],
                SEQ_LEN * 4,
            ),
            "golden_ffn_gate_f32": tensor_desc(
                "ffn_gate_f32.bin",
                "F32",
                32,
                [SEQ_LEN, FFN_DIM],
                FFN_DIM * 4,
            ),
            "golden_ffn_up_f32": tensor_desc(
                "ffn_up_f32.bin",
                "F32",
                32,
                [SEQ_LEN, FFN_DIM],
                FFN_DIM * 4,
            ),
            "golden_ffn_up_row_scale": tensor_desc(
                "ffn_up_row_scale.bin",
                "F32",
                32,
                [SEQ_LEN],
                SEQ_LEN * 4,
            ),
            "golden_ffn_up_q8": tensor_desc(
                "ffn_up_q8.bin",
                "I8",
                8,
                [SEQ_LEN, FFN_DIM],
                FFN_DIM,
            ),
        },
        "attributes": {
            "seq_len": SEQ_LEN,
            "embed_dim": EMBED_DIM,
            "key_dim": KEY_DIM,
            "value_dim": VALUE_DIM,
            "n_head_q": N_HEAD_Q,
            "n_head_kv": N_HEAD_KV,
            "ffn_dim": FFN_DIM,
            "input": "deterministic nonzero input.bin embedded with incbin",
            "weights": "all projection and FFN int8 weights are deterministic nonzero .bin files embedded with incbin",
            "weight_pattern": "64x64 nonzero deterministic pattern repeated over output and K dimensions",
            "expected": "full nonzero-weight llama block output",
        },
        "generator": {
            "tool": "golden/manual/generators/layer/gen_llama_block_nonzero_weight.py",
            "created": "2026-05-19",
        },
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    input_data = input_matrix()
    attn_weight = norm_weight(1)
    ffn_weight = norm_weight(9)

    (OUT_ROOT / "input.bin").write_bytes(input_data.astype("<f4").tobytes())
    for name, (out_dim, in_dim, seed) in WEIGHTS.items():
        emit_weight_bin(name, out_dim, in_dim, seed)

    attn_norm_f32, attn_norm_scale = rmsnorm_with_scale(input_data, attn_weight)
    attn_norm_q8 = smoothquant_stage2(attn_norm_f32, attn_norm_scale)
    emit_i8_bin(OUT_ROOT / "attn_norm_q8.bin", attn_norm_q8)
    emit_f32_bin(OUT_ROOT / "attn_norm_scale.bin", attn_norm_scale)

    q_acc = grouped_matmul_i8(attn_norm_q8, N_HEAD_Q * KEY_DIM, WEIGHTS["proj_q"][2])
    k_acc = grouped_matmul_i8(attn_norm_q8, N_HEAD_KV * KEY_DIM, WEIGHTS["proj_k"][2])
    v_acc = grouped_matmul_i8(attn_norm_q8, N_HEAD_KV * VALUE_DIM, WEIGHTS["proj_v"][2])

    q_bf16 = apply_rope(dequant(q_acc, attn_norm_scale, PROJ_Q_SCALE), N_HEAD_Q * KEY_DIM)
    k_bf16 = apply_rope(dequant(k_acc, attn_norm_scale, PROJ_K_SCALE), N_HEAD_KV * KEY_DIM)
    v_bf16 = to_bf16_f32(dequant(v_acc, attn_norm_scale, PROJ_V_SCALE))
    emit_bf16_bin(OUT_ROOT / "q_bf16.bin", q_bf16)
    emit_bf16_bin(OUT_ROOT / "k_bf16.bin", k_bf16)
    v_bf16_t = np.transpose(
        v_bf16.reshape(SEQ_LEN, N_HEAD_KV, VALUE_DIM),
        (1, 2, 0),
    ).copy()
    emit_bf16_bin(OUT_ROOT / "v_bf16_t.bin", v_bf16_t)

    scores_bf16 = np.empty((N_HEAD_Q, SEQ_LEN, SEQ_LEN), dtype=np.float32)
    attn_context_f32 = np.empty((SEQ_LEN, EMBED_DIM), dtype=np.float32)
    q_heads = q_bf16.reshape(SEQ_LEN, N_HEAD_Q, KEY_DIM)
    k_heads = k_bf16.reshape(SEQ_LEN, N_HEAD_KV, KEY_DIM)
    for h in range(N_HEAD_Q):
        kv_h = h // (N_HEAD_Q // N_HEAD_KV)
        score = f32a(q_heads[:, h, :] @ k_heads[:, kv_h, :].T)
        probs = masked_softmax_bf16(score)
        scores_bf16[h] = probs
        ctx = f32a(probs @ v_bf16_t[kv_h].T)
        attn_context_f32[:, h * VALUE_DIM:(h + 1) * VALUE_DIM] = ctx
    emit_bf16_bin(OUT_ROOT / "scores_head0_bf16.bin", scores_bf16[0])

    attn_q8, attn_scale = smoothquant(attn_context_f32)
    proj_o_acc = grouped_matmul_i8(attn_q8, EMBED_DIM, WEIGHTS["proj_o"][2])
    proj_o_deq = dequant(proj_o_acc, attn_scale, PROJ_O_SCALE)
    proj_o_f32 = f32a(proj_o_deq + input_data)
    emit_f32_bin(OUT_ROOT / "proj_o_f32.bin", proj_o_f32)

    ffn_norm_f32, ffn_norm_scale = rmsnorm_with_scale(proj_o_f32, ffn_weight)
    ffn_norm_q8 = smoothquant_stage2(ffn_norm_f32, ffn_norm_scale)
    emit_i8_bin(OUT_ROOT / "ffn_norm_q8.bin", ffn_norm_q8)
    emit_f32_bin(OUT_ROOT / "ffn_norm_scale.bin", ffn_norm_scale)

    gate_acc = grouped_matmul_i8(ffn_norm_q8, FFN_DIM, WEIGHTS["ffn_gate"][2])
    up_acc = grouped_matmul_i8(ffn_norm_q8, FFN_DIM, WEIGHTS["ffn_up"][2])
    ffn_gate_f32 = silu(dequant(gate_acc, ffn_norm_scale, FFN_GATE_SCALE))
    ffn_up_f32 = f32a(dequant(up_acc, ffn_norm_scale, FFN_UP_SCALE) * ffn_gate_f32)
    ffn_up_scale = f32a(np.max(np.abs(ffn_up_f32), axis=1) / np.float32(127.0))
    ffn_up_q8 = smoothquant_stage2(ffn_up_f32, ffn_up_scale)
    emit_f32_bin(OUT_ROOT / "ffn_gate_f32.bin", ffn_gate_f32)
    emit_f32_bin(OUT_ROOT / "ffn_up_f32.bin", ffn_up_f32)
    emit_f32_bin(OUT_ROOT / "ffn_up_row_scale.bin", ffn_up_scale)
    emit_i8_bin(OUT_ROOT / "ffn_up_q8.bin", ffn_up_q8)

    down_acc = grouped_matmul_i8(ffn_up_q8, EMBED_DIM, WEIGHTS["ffn_down"][2])
    down_deq = dequant(down_acc, ffn_up_scale, FFN_DOWN_SCALE)
    output = f32a(down_deq + proj_o_f32)

    emit_f32_bin(OUT_ROOT / "golden_output.bin", output)
    emit_header(attn_weight, ffn_weight)
    emit_manifest()

    print(f"generated {CASE_ID}")
    print(f"  output min={float(np.min(output)):.8g} max={float(np.max(output)):.8g}")
    print(f"  output_delta max={float(np.max(np.abs(output - input_data))):.8g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
