#!/usr/bin/env python3
"""Generate manual golden data for matmul + vector post-op fusion cases."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path


M = 128
N = 128
TENSOR_GOLDEN = Path("golden/manual/tensor/matmul_i8_128_128_128_zeroinit/golden.bin")
OUT_ROOT = Path("golden/manual/fusion")
WEIGHT_SCALE = 0.001


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def bits_to_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def input_scale(row: int) -> float:
    return f32(0.001 + (row % 17) * 0.00001)


def residual_value(row: int, col: int) -> float:
    v = ((row * 13 + col * 7) % 257) - 128
    return f32(v * 0.00025)


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


def silu(x: float) -> float:
    return f32(x / f32(1.0 + exp_approx(f32(-x))))


def read_matmul() -> list[int]:
    data = TENSOR_GOLDEN.read_bytes()
    return list(struct.unpack(f"<{M * N}i", data))


def write_manifest(case_dir: Path, case_id: str, dtype: str, element_bits: int) -> None:
    elem_bytes = element_bits // 8
    manifest = {
        "id": case_id,
        "op": case_id.rsplit("_m", 1)[0],
        "level": "fusion",
        "tensors": {
            "golden_output": {
                "path": "golden_output.bin",
                "dtype": dtype,
                "element_bits": element_bits,
                "total_bytes": M * N * elem_bytes,
                "layout": "row_major",
                "shape": [M, N],
                "stride_bytes": N * elem_bytes,
            }
        },
        "attributes": {
            "m": M,
            "n": N,
            "k": 128,
            "input_scale": "0.001f + (row % 17) * 0.00001f",
            "weight_scale": WEIGHT_SCALE,
        },
        "generator": {
            "tool": "golden/manual/generators/fusion/gen_matmul_post.py",
            "source": str(TENSOR_GOLDEN),
            "created": "2026-05-18",
        },
    }
    (case_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def emit_f32(case_id: str, values: list[float]) -> None:
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "golden_output.bin").write_bytes(
        struct.pack(f"<{len(values)}f", *[f32(v) for v in values])
    )
    write_manifest(case_dir, case_id, "F32", 32)


def emit_f16(case_id: str, values: list[float]) -> None:
    case_dir = OUT_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "golden_output.bin").write_bytes(
        b"".join(struct.pack("<e", f32(v)) for v in values)
    )
    write_manifest(case_dir, case_id, "F16", 16)


def main() -> int:
    matmul = read_matmul()
    deq = [dequant(acc, row=i // N) for i, acc in enumerate(matmul)]

    emit_f32(
        "matmul_dequant_silu_m128_n128",
        [silu(v) for v in deq],
    )
    emit_f32(
        "matmul_dequant_resadd_m128_n128",
        [
            f32(v + residual_value(i // N, i % N))
            for i, v in enumerate(deq)
        ],
    )
    emit_f16("matmul_dequant_bf16cvt_m128_n128", deq)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
