#!/usr/bin/env python3
"""Deterministic nonzero fixture and independent unbounded-integer oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def generate(output: Path, platform_path: Path):
    output.mkdir(parents=True, exist_ok=True)
    platform = json.loads(platform_path.read_text())
    tcm = platform.get('tcm') or {'client_base': 0, 'bytes': 0}
    (output / "system_platform.h").write_text(
        "/* Generated from checked target bundle platform.json. */\n"
        f"#define CUTE_SYSTEM_TCM_BASE {hex(int(tcm['client_base']))}\n"
        f"#define CUTE_SYSTEM_TCM_BYTES {tcm['bytes']}\n"
        f"#define CUTE_SYSTEM_VLEN {platform['vlen']}\n"
        f"#define CUTE_SYSTEM_DRAM_BASE {hex(platform['dram_base'])}ULL\n"
        f"#define CUTE_SYSTEM_DRAM_BYTES {platform['dram_bytes']}ULL\n"
        f"#define CUTE_SYSTEM_PROGRAM_RAM_END {hex(platform['program_ram']['end'])}ULL\n"
        f"#define CUTE_SYSTEM_HEAP_BYTES {platform['heap_bytes']}\n")
    n = 128
    a = [(13 * i + 7 * k + 3) % 17 - 8 for i in range(n) for k in range(n)]
    b = [(5 * j + 11 * k + 1) % 19 - 9 for j in range(n) for k in range(n)]
    result = [sum(a[i*n+k] * b[j*n+k] for k in range(n))
              for i in range(n) for j in range(n)]
    assert any(result) and min(a) < 0 < max(a) and min(b) < 0 < max(b)
    header = ["/* Generated deterministic P0/T1 fixture. */", "#include <stdint.h>"]
    for name, values in [("sys_input_a", a), ("sys_input_b", b)]:
        header.append(f"const int8_t {name}[16384] __attribute__((aligned(64))) = {{")
        header.extend(",".join(str(v) for v in values[i:i+64]) + ","
                      for i in range(0, len(values), 64))
        header.append("};")
    (output / "system_inputs.h").write_text("\n".join(header) + "\n")
    files = {"a.bin": bytes(v & 255 for v in a),
             "a_line.bin": bytes(v & 255 for v in a[:64]),
             "b.bin": bytes(v & 255 for v in b),
             "c.bin": b"".join(struct.pack("<I", v & 0xffffffff) for v in result)}
    for name, data in files.items():
        (output / name).write_bytes(data)
    (output / "fixture.json").write_text(json.dumps({
        "version": 1, "shape": [n, n, n], "dtype": "i8_i8_i32", "platform": platform,
        "b_layout": "N,K", "oracle": "Python unbounded-integer dot products, wrap to u32",
        "sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                   for name in [*files, "system_inputs.h", "system_platform.h"]}
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--platform", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.platform)
