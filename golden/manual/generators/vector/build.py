#!/usr/bin/env python3
"""
Phase C0 RVV golden generator.
Reads cases.yaml, compiles RISC-V binaries, runs on QEMU.
Outputs per-case directory with .h, .bin files, and manifest.json.

Usage:
    python3 build.py --case silu
    python3 build.py --all
"""

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml not installed. Run: pip install pyyaml")

SDK_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
GEN_ROOT = Path(__file__).resolve().parent
CASES_DIR = GEN_ROOT / "cases"
BUILD_DIR = SDK_ROOT / "build" / "golden" / "manual" / "vector"
OUTPUT_DIR = SDK_ROOT / "golden" / "manual" / "vector"
YAML_PATH = GEN_ROOT / "cases.yaml"


def load_config():
    with open(YAML_PATH, "r") as f:
        return yaml.safe_load(f)


def resolve_tool(cfg, key):
    p = Path(cfg["toolchain"][key])
    if not p.is_absolute():
        p = SDK_ROOT / p
    return str(p)


def make_case_id(name, params):
    id_parts = [name]
    for k, v in params.items():
        if k.startswith("DEQUANT") or k.startswith("RMSNORM") or k.startswith("FUSE") or k == "ROPE_POS":
            continue
        dim = k.replace("GOLDEN_", "").lower()
        id_parts.append(f"{dim}{v}")
    return "_".join(id_parts)


def make_test_case_id(golden_case_id):
    return f"primitive_{golden_case_id}"


def is_bf16_output_case(case_name):
    return (
        "bf16cvt" in case_name
        or case_name.startswith("rope_pos")
        or case_name == "masked_softmax"
    )


# ---------------------------------------------------------------------------
# Parse .h content and extract arrays → .bin files
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "float":    {"dtype": "F32", "element_bits": 32, "struct_fmt": "<f"},
    "uint16_t": {"dtype": "F16", "element_bits": 16, "struct_fmt": "<H"},
    "int32_t":  {"dtype": "I32", "element_bits": 32, "struct_fmt": "<i"},
    "int8_t":   {"dtype": "I8",  "element_bits":  8, "struct_fmt": "<b"},
    "uint8_t":  {"dtype": "U1_PACKED", "element_bits": 8, "struct_fmt": "<B"},
}


def parse_arrays(h_content):
    """Parse all static const arrays from .h content."""
    # Match: static const <type> <name>[<size>] = { <values> };
    pattern = re.compile(
        r'static\s+const\s+(\w+)\s+(\w+)\s*\[\d*\]\s*=\s*\{([^}]+)\};',
        re.DOTALL,
    )
    arrays = {}
    for m in pattern.finditer(h_content):
        ctype = m.group(1)
        name = m.group(2)
        values_str = m.group(3)
        if ctype not in DTYPE_MAP:
            continue
        info = DTYPE_MAP[ctype]
        # Parse values
        raw = [v.strip() for v in values_str.split(",") if v.strip()]
        if ctype == "float":
            values = [float.fromhex(v) for v in raw]
        elif ctype == "uint16_t":
            values = [int(v, 16) for v in raw]
        elif ctype == "int8_t":
            values = [int(v) for v in raw]
        elif ctype == "int32_t":
            values = [int(v) for v in raw]
        elif ctype == "uint8_t":
            values = [int(v, 16) for v in raw]
        else:
            continue
        arrays[name] = {
            "dtype": info["dtype"],
            "element_bits": info["element_bits"],
            "struct_fmt": info["struct_fmt"],
            "values": values,
            "count": len(values),
        }
    return arrays


def write_bin_files(case_dir, arrays):
    """Write .bin files for each parsed array. Returns tensor metadata for manifest."""
    tensors = {}
    for name, info in arrays.items():
        bin_path = case_dir / f"{name}.bin"
        fmt = f'<{info["count"]}{info["struct_fmt"][-1]}'
        data = struct.pack(fmt, *info["values"])
        with open(bin_path, "wb") as f:
            f.write(data)
        total_bytes = len(data)
        element_bytes = info["element_bits"] // 8
        tensors[name] = {
            "path": bin_path.name,
            "dtype": info["dtype"],
            "element_bits": info["element_bits"],
            "total_bytes": total_bytes,
        }
    return tensors


def parse_defines(h_content):
    """Extract #define values from .h content."""
    defs = {}
    for m in re.finditer(r'#define\s+(GOLDEN_\w+)\s+(\d+)', h_content):
        defs[m.group(1)] = int(m.group(2))
    return defs


def infer_shape(name, params, count):
    """Infer shape from array name and YAML params."""
    name_lower = name.lower()
    m = params.get("GOLDEN_M", 0)
    n = params.get("GOLDEN_N", 0)
    k = params.get("GOLDEN_K", 0)
    batch = params.get("GOLDEN_BATCH", 0)
    seq = params.get("GOLDEN_SEQ_LEN", 0)
    hidden = params.get("GOLDEN_HIDDEN_DIM", 0)

    # 1D tensors
    if "scale" in name_lower and count > 0:
        if m and count == m:
            return [m]
        if batch and seq and count == batch * seq:
            return [batch, seq]
    if "row_absmax" in name_lower and m and count == m:
        return [m]
    if "mask" in name_lower:
        return [count]
    if "theta" in name_lower:
        return [count]
    if "weight_scale" in name_lower and count == 1:
        return [1]

    # 3D: BATCH*SEQ_LEN*HIDDEN_DIM
    if batch and seq and hidden and count == batch * seq * hidden:
        return [batch, seq, hidden]

    # 2D: M*N or M*K
    if m and n and count == m * n:
        return [m, n]
    if m and k and count == m * k:
        return [m, k]
    if n and count == n:
        return [n]

    return [count]


def write_manifest(case_dir, case_id, case_name, params, tensors, h_filename):
    """Generate manifest.json."""
    from datetime import date

    cfg = load_config()
    # Enrich tensor entries with shape and layout
    for name, t in tensors.items():
        t["layout"] = "row_major"
        t["shape"] = infer_shape(name, params, t["total_bytes"] // (t["element_bits"] // 8))
        if len(t["shape"]) >= 2:
            t["stride_bytes"] = t["shape"][-1] * (t["element_bits"] // 8)
        else:
            t["stride_bytes"] = t["total_bytes"]

    manifest = {
        "id": case_id,
        "op": case_name,
        "level": "primitive",
        "op_ref": f"ops/vector/{case_name}.yaml",
        "tensors": tensors,
        "attributes": {k.replace("GOLDEN_", "").lower(): v for k, v in params.items()
                       if not k.startswith("DEQUANT") and not k.startswith("RMSNORM")},
        "verify": {
            "mode": "bit_exact",
            "tensors": [name for name in tensors if "golden" in name.lower() or "output" in name.lower()],
        },
        "generator": {
            "tool": "rvv_qemu",
            "source": "NVWA/llama3.2_1B/data_flow/gloden_opt.h",
            "qemu": cfg["toolchain"]["qemu"],
            "qemu_cpu": cfg["toolchain"]["qemu_cpu"],
            "compiler": str(Path(cfg["toolchain"]["cc"]).name),
            "cflags": cfg["toolchain"]["cflags"],
            "created": date.today().isoformat(),
        },
        "files": {
            "header": h_filename,
        },
    }

    manifest_path = case_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def write_test_case(test_case_id, case_name, golden_manifest_rel):
    """Create tests/primitive/<test_case_id>/case.json referencing the golden manifest."""
    test_dir = SDK_ROOT / "tests" / "primitive" / test_case_id
    test_dir.mkdir(parents=True, exist_ok=True)
    case_path = test_dir / "case.json"
    if case_path.exists():
        with open(case_path, "r") as f:
            case = json.load(f)
        case["golden"] = str(golden_manifest_rel)
        case.setdefault("id", test_case_id)
        case.setdefault("op_ref", f"ops/vector/{case_name}.yaml")
        case.setdefault("level", "primitive")
    else:
        case = {
            "id": test_case_id,
            "op_ref": f"ops/vector/{case_name}.yaml",
            "level": "primitive",
            "build": {
                "source": "test.c",
                "target": "test.riscv",
            },
            "run": {
                "hwconfig": "cute4tops_shuttle512_d512_v512_m512_sysbus512_membus1_core_dramsim48",
                "trace_source": "run.out",
            },
            "golden": str(golden_manifest_rel),
            "verify": {
                "mode": "return_code",
            },
        }
    with open(case_path, "w") as f:
        json.dump(case, f, indent=2)
        f.write("\n")
    return case_path


# ---------------------------------------------------------------------------
# Compile & Run
# ---------------------------------------------------------------------------

def compile_case(cc, cflags, include_dir, src, params, output_binary):
    output_binary.parent.mkdir(parents=True, exist_ok=True)
    defines = [f"-D{k}={v}" for k, v in params.items()]
    cmd = [cc, *cflags.split(), f"-I{include_dir}", *defines, str(src),
           "-o", str(output_binary), "-lm"]
    print(f"  compiling {src.name} -> {output_binary.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: compile failed", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return output_binary


def run_case(qemu, qemu_cpu, binary, case_dir, case_id, case_name, params):
    case_dir.mkdir(parents=True, exist_ok=True)
    h_filename = f"golden_{case_name}.h"
    h_path = case_dir / h_filename

    cmd = [qemu, "-cpu", qemu_cpu, str(binary)]
    print(f"  running {binary.name} -> {case_dir.name}/")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"ERROR: QEMU run failed", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Save .h
    with open(h_path, "w") as f:
        f.write(result.stdout)
    print(f"    {h_filename} ({len(result.stdout)} bytes)")

    # Parse and write .bin
    arrays = parse_arrays(result.stdout)
    if not arrays:
        print(f"    WARNING: no arrays found in output", file=sys.stderr)
        return

    tensor_meta = write_bin_files(case_dir, arrays)
    if is_bf16_output_case(case_name):
        for name, tensor in tensor_meta.items():
            if "output" in name.lower() and tensor["element_bits"] == 16:
                tensor["dtype"] = "BF16"
    for name in tensor_meta:
        print(f"    {name}.bin")

    # Write manifest.json
    manifest_path = write_manifest(case_dir, case_id, case_name, params, tensor_meta, h_filename)
    print(f"    manifest.json")

    # Write tests/primitive/<case_id>/case.json
    golden_rel = manifest_path.relative_to(SDK_ROOT)
    test_case_path = write_test_case(make_test_case_id(case_id), case_name, golden_rel)
    print(f"    test -> {test_case_path.relative_to(SDK_ROOT)}")


def build_case(cfg, case_name):
    cases = cfg["cases"]
    case = next((c for c in cases if c["name"] == case_name), None)
    if case is None:
        names = [c["name"] for c in cases]
        print(f"ERROR: unknown case '{case_name}'. Available: {', '.join(names)}", file=sys.stderr)
        sys.exit(1)

    params = case.get("params", {})
    case_id = make_case_id(case_name, params)

    cc = resolve_tool(cfg, "cc")
    cflags = cfg["toolchain"]["cflags"]
    qemu = resolve_tool(cfg, "qemu")
    qemu_cpu = cfg["toolchain"]["qemu_cpu"]
    include_dir = GEN_ROOT / "include"
    src = CASES_DIR / case["source"]
    binary = BUILD_DIR / case_id / f"gen_{case_name}.riscv"
    case_dir = OUTPUT_DIR / case_id

    print(f"[{case_name}] building...")
    compile_case(cc, cflags, include_dir, src, params, binary)
    run_case(qemu, qemu_cpu, binary, case_dir, case_id, case_name, params)
    print(f"[{case_name}] done.")


def main():
    parser = argparse.ArgumentParser(description="Phase C0 RVV golden generator")
    parser.add_argument("--case", help="Build a single case")
    parser.add_argument("--all", action="store_true", help="Build all cases")
    args = parser.parse_args()

    cfg = load_config()

    if args.case:
        build_case(cfg, args.case)
    elif args.all:
        for case in cfg["cases"]:
            build_case(cfg, case["name"])
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
