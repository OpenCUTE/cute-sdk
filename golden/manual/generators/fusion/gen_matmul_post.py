#!/usr/bin/env python3
"""Generate manual golden data for matmul + vector post-op fusion cases."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path


M128 = 128
N128 = 128
N64 = 64
K64 = 64
K128 = 128
TENSOR_GOLDEN = Path("golden/manual/tensor/matmul_i8_128_128_128_zeroinit/golden.bin")
OUT_ROOT = Path("golden/manual/fusion")
WEIGHT_SCALE = 0.001


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f16(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", f32(value)))[0]


def pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *[f32(v) for v in values])


def pack_f16(values: list[float]) -> bytes:
    return b"".join(struct.pack("<e", f32(v)) for v in values)


def bits_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def input_scale(row: int) -> float:
    return f32(0.001 + (row % 17) * 0.00001)


def residual_value(row: int, col: int) -> float:
    v = ((row * 13 + col * 7) % 257) - 128
    return f32(v * 0.00025)


def hadamard_lhs_value(row: int, col: int) -> float:
    v = ((row * 11 + col * 5) % 127) - 63
    return f32(v * 0.03125)


def softmax_a_value(row: int, col: int) -> float:
    v = ((row * 5 + col * 3) % 17) - 8
    return f16(v * 0.0625)


def softmax_b_value(row: int, col: int) -> float:
    v = ((row * 7 + col * 11) % 19) - 9
    return f16(v * 0.0625)


def attention_score_value(row: int, col: int) -> float:
    if col > row:
        return f16(0.0)
    v = ((row * 3 + col * 5) % 17) + 1
    return f16(v * 0.00390625)


def attention_value_value(row: int, col: int) -> float:
    v = ((row * 13 + col * 7) % 31) - 15
    return f16(v * 0.03125)


def dequant(acc: int, row: int) -> float:
    scale = f32(input_scale(row) * f32(WEIGHT_SCALE))
    return f32(f32(acc) * scale)


def exp_approx(x: float) -> float:
    neg_ln2 = f32(-0.69314718056)
    inv_ln2 = f32(1.44269504089)
    af = f32(x * inv_ln2)
    magic = f32(float.fromhex("0x1.8p23"))
    a = f32(f32(af + magic) - magic)
    a_int = int(a)
    if a_int > 127:
        return math.inf
    if a_int < -126:
        return 0.0

    a2 = bits_to_f32((a_int + 127) << 23)
    b = f32(x + f32(neg_ln2 * a))
    c0 = f32(1.0)
    c1 = f32(1.0)
    c2 = f32(0.5)
    c3 = f32(0.166666666667)
    c4 = f32(0.041666666667)
    c5 = f32(0.008333333333)
    c6 = f32(0.001388888889)

    p = f32(c5 + f32(c6 * b))
    p = f32(c4 + f32(p * b))
    p = f32(c3 + f32(p * b))
    p = f32(c2 + f32(p * b))
    p = f32(c1 + f32(p * b))
    p = f32(c0 + f32(p * b))
    return f32(a2 * p)


def round_magic(value: float) -> float:
    magic = f32(float.fromhex("0x1.8p23"))
    return f32(f32(value + magic) - magic)


def sin_small_approx(x: float) -> float:
    x2 = f32(x * x)
    c1 = f32(1.0)
    c3 = f32(-0.166666666667)
    c5 = f32(0.008333333333)
    c7 = f32(-0.0001984126984)
    c9 = f32(0.000002755731922)
    c11 = f32(-0.000000025052108)

    result = f32(c9 + f32(c11 * x2))
    result = f32(c7 + f32(result * x2))
    result = f32(c5 + f32(result * x2))
    result = f32(c3 + f32(result * x2))
    result = f32(c1 + f32(result * x2))
    return f32(result * x)


def sin_approx(x: float) -> float:
    pi = f32(3.14159265359)
    pi_div_2 = f32(pi / 2.0)
    inv_pi = f32(0.31830988618)

    new_rad = f32(x + pi_div_2)
    round_v = round_magic(f32(new_rad * inv_pi))
    new_rad = f32(new_rad - f32(round_v * pi))
    new_rad = f32(new_rad - pi_div_2)

    result = sin_small_approx(new_rad)
    round_int = int(round_v)
    if round_int & 1:
        result = f32(-result)
    return result


def cos_approx(x: float) -> float:
    pi = f32(3.14159265359)
    pi_div_2 = f32(pi / 2.0)
    return sin_approx(f32(x + pi_div_2))


def silu(x: float) -> float:
    return f32(x / f32(1.0 + exp_approx(f32(-x))))


def rope_theta(k: int, half_dim: int) -> float:
    head_dim = half_dim * 2
    return f32(math.pow(10000.0, -2.0 * float(k) / float(head_dim)))


def rope(values: list[float], rows: int, cols: int, pos: int) -> list[float]:
    half_dim = cols // 2
    output = [0.0] * (rows * cols)
    for row in range(rows):
        pos_r = row + pos
        base = row * cols
        for k in range(half_dim):
            theta = rope_theta(k, half_dim)
            angle = f32(theta * f32(pos_r))
            sin_v = sin_approx(angle)
            cos_v = cos_approx(angle)
            real_in = values[base + 2 * k]
            imag_in = values[base + 2 * k + 1]
            output[base + k] = f32(f32(real_in * cos_v) - f32(imag_in * sin_v))
            output[base + half_dim + k] = f32(
                f32(real_in * sin_v) + f32(imag_in * cos_v)
            )
    return output


def read_matmul() -> list[int]:
    data = TENSOR_GOLDEN.read_bytes()
    return list(struct.unpack(f"<{M128 * N128}i", data))


def slice_cols(values: list[int], rows: int, src_cols: int, dst_cols: int) -> list[int]:
    sliced: list[int] = []
    for row in range(rows):
        start = row * src_cols
        sliced.extend(values[start:start + dst_cols])
    return sliced


def write_manifest(case_dir: Path, case_id: str, tensors: dict, attrs: dict) -> None:
    manifest = {
        "id": case_id,
        "op": case_id.rsplit("_m", 1)[0],
        "level": "fusion",
        "tensors": tensors,
        "attributes": attrs,
        "generator": {
            "tool": "golden/manual/generators/fusion/gen_matmul_post.py",
            "source": str(TENSOR_GOLDEN),
            "created": "2026-05-18",
        },
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def tensor_desc(path: str, dtype: str, element_bits: int,
                shape: list[int], stride_bytes: int) -> dict:
    elem_bytes = element_bits // 8
    total = elem_bytes
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


def emit_single(case_id: str, values: list[float],
                rows: int, cols: int, dtype: str, element_bits: int) -> None:
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = pack_f32(values) if element_bits == 32 else pack_f16(values)
    (case_dir / "golden_output.bin").write_bytes(payload)
    elem_bytes = element_bits // 8
    write_manifest(
        case_dir,
        case_id,
        {
            "golden_output": tensor_desc(
                "golden_output.bin",
                dtype,
                element_bits,
                [rows, cols],
                cols * elem_bytes,
            )
        },
        {
            "m": rows,
            "n": cols,
            "k": 128,
            "input_scale": "0.001f + (row % 17) * 0.00001f",
            "weight_scale": WEIGHT_SCALE,
        },
    )


def emit_hadamard(case_id: str, output: list[float], row_absmax: list[float]) -> None:
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "golden_output.bin").write_bytes(pack_f32(output))
    (case_dir / "golden_row_absmax.bin").write_bytes(pack_f32(row_absmax))
    write_manifest(
        case_dir,
        case_id,
        {
            "golden_output": tensor_desc(
                "golden_output.bin", "F32", 32, [M128, N128], N128 * 4
            ),
            "golden_row_absmax": tensor_desc(
                "golden_row_absmax.bin", "F32", 32, [M128], M128 * 4
            ),
        },
        {
            "m": M128,
            "n": N128,
            "k": 128,
            "input_scale": "0.001f + (row % 17) * 0.00001f",
            "weight_scale": WEIGHT_SCALE,
            "lhs": "(((row * 11 + col * 5) % 127) - 63) * 0.03125f",
        },
    )


def emit_attention_context(case_id: str, output: list[float]) -> None:
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "golden_output.bin").write_bytes(pack_f32(output))
    write_manifest(
        case_dir,
        case_id,
        {
            "golden_output": tensor_desc(
                "golden_output.bin", "F32", 32, [M128, N64], N64 * 4
            )
        },
        {
            "m": M128,
            "n": N64,
            "k": K128,
            "input_dtype": "F16F16F32",
            "scores": "causal lower triangle f16 deterministic values",
            "value": "f16 deterministic values, B layout [N][K]",
        },
    )


def emit_softmax(case_id: str, output: list[float],
                 rows: int, cols: int, k: int) -> None:
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "golden_output_f16.bin").write_bytes(pack_f16(output))
    write_manifest(
        case_dir,
        case_id,
        {
            "golden_output_f16": tensor_desc(
                "golden_output_f16.bin", "F16", 16, [rows, cols], cols * 2
            )
        },
        {
            "m": rows,
            "n": cols,
            "k": k,
            "kv_scale": 0.125,
            "mask": f"causal, max_ctx_len={cols}",
            "input_dtype": "F16F16F32",
        },
    )


def softmax_scores(rows: int, cols: int, kdim: int) -> list[float]:
    scores = []
    for row in range(rows):
        for col in range(cols):
            acc = f32(0.0)
            for k in range(kdim):
                prod = f32(softmax_a_value(row, k) * softmax_b_value(col, k))
                acc = f32(acc + prod)
            scores.append(acc)
    return scores


def masked_softmax(scores: list[float], rows: int, cols: int, scale: float) -> list[float]:
    output = [0.0] * (rows * cols)
    for row in range(rows):
        base = row * cols
        masked = []
        for col in range(cols):
            value = f32(scores[base + col] * scale)
            masked.append(value if col <= row else -math.inf)
        max_val = max(masked)
        exps = [
            0.0 if math.isinf(v) and v < 0 else exp_approx(f32(v - max_val))
            for v in masked
        ]
        denom = f32(sum(exps))
        for col in range(cols):
            output[base + col] = f32(exps[col] / denom)
    return output


def attention_context() -> list[float]:
    output = []
    for row in range(M128):
        for col in range(N64):
            acc = f32(0.0)
            for k in range(K128):
                prod = f32(attention_score_value(row, k) *
                           attention_value_value(col, k))
                acc = f32(acc + prod)
            output.append(acc)
    return output


def main() -> int:
    matmul = read_matmul()
    deq128 = [dequant(acc, row=i // N128) for i, acc in enumerate(matmul)]

    emit_single(
        "matmul_dequant_silu_m128_n128",
        [silu(v) for v in deq128],
        M128,
        N128,
        "F32",
        32,
    )
    emit_single(
        "matmul_dequant_resadd_m128_n128",
        [
            f32(v + residual_value(i // N128, i % N128))
            for i, v in enumerate(deq128)
        ],
        M128,
        N128,
        "F32",
        32,
    )
    emit_single("matmul_dequant_bf16cvt_m128_n128", deq128, M128, N128, "F16", 16)
    emit_single(
        "matmul_dequant_bf16cvt_transpose_m128_n128",
        [
            dequant(matmul[col * N128 + row], row=col)
            for row in range(N128)
            for col in range(M128)
        ],
        N128,
        M128,
        "F16",
        16,
    )

    matmul64 = slice_cols(matmul, M128, N128, N64)
    deq64 = [dequant(acc, row=i // N64) for i, acc in enumerate(matmul64)]
    emit_single(
        "matmul_dequant_rope_bf16cvt_m128_n64",
        rope(deq64, M128, N64, 17),
        M128,
        N64,
        "F16",
        16,
    )

    hadamard_output: list[float] = []
    row_absmax = [0.0] * M128
    for row in range(M128):
        for col in range(N128):
            value = f32(deq128[row * N128 + col] * hadamard_lhs_value(row, col))
            hadamard_output.append(value)
            row_absmax[row] = max(row_absmax[row], abs(value))
        row_absmax[row] = f32(row_absmax[row])
    emit_hadamard("matmul_dequant_hadamard_m128_n128", hadamard_output, row_absmax)

    emit_softmax(
        "matmul_masked_softmax_kvscale_bf16cvt_m128_n64_k64",
        masked_softmax(softmax_scores(M128, N64, K64), M128, N64, 0.125),
        M128,
        N64,
        K64,
    )
    emit_softmax(
        "matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64",
        masked_softmax(softmax_scores(M128, N128, K64), M128, N128, 0.125),
        M128,
        N128,
        K64,
    )
    emit_attention_context(
        "attention_context_f16_m128_n64_k128",
        attention_context(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
