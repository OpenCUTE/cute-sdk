#!/usr/bin/env python3
"""Build, simulate, and verify parameterized SV FPE GEMM runtime cases.

This is the SDK replacement for cutetest/sv_fpe_datatype_mm_test/Makefile.
The C-model remains in CUTE/cute-fpe; this tool owns the SDK-side test data,
RISC-V build, simulator invocation, and memverify comparison.
"""

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path


SDK_ROOT = Path(__file__).resolve().parents[1]
CUTE_ROOT = SDK_ROOT.parent
CASE_DIR = SDK_ROOT / "tests" / "runtime" / "sv_fpe_datatype_mm"
GENERATED_DIR = CASE_DIR / "generated"
BUILD_DIR = SDK_ROOT / "build"
HOST_DIR = BUILD_DIR / "host"
GENERATOR_SOURCE = CASE_DIR / "generate_gemm_golden.cpp"
GENERATOR_BINARY = HOST_DIR / "generate_sv_fpe_gemm"
SIMULATOR_SCRIPT = CUTE_ROOT / "scripts" / "run-simulator-test.sh"
IMPORT_HEADER = SDK_ROOT / "memverify" / "tools" / "import_header_golden.py"

CONFIG_WIDTHS = {
    "CUTE2TopsSCP64CMLDebugConfig": 32,
    "CUTE4TopsSCP64CMLDebugConfig": 64,
}

BATCH_ORDERS = {
    0: [0, 1, 2, 3, 4, 5, 6, 7],
    1: [1],
    2: [0, 1],
    3: [5, 1],
    4: [3, 1],
    5: [2, 1],
}


def run(command, *, cwd=None, env=None):
    print("+", " ".join(str(arg) for arg in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_capture(command, *, cwd=None, env=None) -> str:
    print("+", " ".join(str(arg) for arg in command), flush=True)
    return subprocess.check_output(command, cwd=cwd, env=env, text=True)


def define_value(header: Path, name: str) -> int:
    match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", header.read_text(), re.MULTILINE)
    if not match:
        raise RuntimeError(f"{header}: missing #define {name}")
    return int(match.group(1))


def ensure_generator() -> None:
    HOST_DIR.mkdir(parents=True, exist_ok=True)
    cmodel_include = CUTE_ROOT / "cute-fpe" / "cmodel" / "necessary_source"
    if (not GENERATOR_BINARY.exists() or
            GENERATOR_BINARY.stat().st_mtime < GENERATOR_SOURCE.stat().st_mtime):
        run([
            "c++", "-x", "c++", "-std=c++11", "-O2",
            "-DNVIDIA_ON=0", "-DFMAC_DUMP_OUTPUT=0",
            f"-I{cmodel_include}", str(GENERATOR_SOURCE),
            "-o", str(GENERATOR_BINARY),
        ])


def generator_args(args, output: Path, *, dtype: int | None = None,
                   bias: str | None = None, m: int | None = None,
                   n: int | None = None, transpose: int | None = None):
    command = [
        str(GENERATOR_BINARY), "--output", str(output),
        "--cute-dtype", str(args.dtype if dtype is None else dtype),
        "--seed", str(args.seed),
        "--fpe-bytes", str(args.fpe_bytes),
        "--bias", args.bias if bias is None else bias,
        "--m", str(args.m if m is None else m),
        "--n", str(args.n if n is None else n),
        "--transpose", str(args.transpose if transpose is None else transpose),
        "--tensor-k-bytes", str(args.tensor_k_bytes),
        "--scale-tile-m", str(args.scale_tile_m),
        "--scale-tile-n", str(args.scale_tile_n),
    ]
    if args.k:
        command += ["--k", str(args.k)]
    return command


def prepare_single_case(args, result_dir: Path) -> tuple[Path, Path]:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    header = GENERATED_DIR / "case_data.h"
    run(generator_args(args, header))
    saved_header = result_dir / "case_data.h"
    shutil.copy2(header, saved_header)
    manifest_dir = result_dir / "golden"
    run([sys.executable, str(IMPORT_HEADER), "--header", str(saved_header),
         "--output", str(manifest_dir)], cwd=SDK_ROOT)
    normalize_manifest(manifest_dir / "manifest.json", saved_header)
    return header, manifest_dir / "manifest.json"


def normalize_manifest(manifest: Path, header: Path) -> None:
    """Make imported header golden match CUTE's physical D write layout."""
    metadata = json.loads(manifest.read_text())
    m = define_value(header, "APPLICATION_M")
    n = define_value(header, "APPLICATION_N")
    transpose = define_value(header, "TRANSPOSE_RESULT")
    d = metadata["tensors"]["D"]
    if transpose:
        golden = manifest.parent / d["path"]
        words = [golden.read_bytes()[offset:offset + 4]
                 for offset in range(0, golden.stat().st_size, 4)]
        physical = bytearray()
        for column in range(n):
            for row in range(m):
                physical.extend(words[row * n + column])
        golden.write_bytes(physical)
        d["shape"] = [n, m]
        d["stride_bytes"] = m * 4
        d["layout"] = "row_major_physical_transpose"
    metadata["defines"]["fpe_bytes"] = define_value(header, "FPE_BYTES")
    metadata["defines"]["cmodel_dtype"] = define_value(header, "CMODEL_DTYPE")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")


def configure_and_build(target: str, args) -> Path:
    run([
        "cmake", "-S", str(SDK_ROOT), "-B", str(BUILD_DIR),
        f"-DCMAKE_TOOLCHAIN_FILE={SDK_ROOT / 'cmake' / 'riscv-toolchain.cmake'}",
        f"-DCUTE_ROOT={CUTE_ROOT}",
        f"-DSV_FPE_BATCH_MODE={args.batch_mode}",
    ])
    run(["cmake", "--build", str(BUILD_DIR), "--target", target,
         "-j", str(args.jobs)])
    elf = BUILD_DIR / "runtime" / f"{target}.riscv"
    if not elf.is_file():
        raise RuntimeError(f"CMake did not produce {elf}")
    return elf


def simulator_trace(config: str, elf: Path, result_dir: Path) -> tuple[Path, Path]:
    run([str(SIMULATOR_SCRIPT), config, str(elf)], cwd=CUTE_ROOT)
    name = elf.stem
    trace_dir = (CUTE_ROOT / "build" / "chipyard" / "debug-logs" /
                 f"chipyard.harness.TestHarness.{config}")
    trace = trace_dir / f"{name}.out"
    uart = trace_dir / f"{name}.log"
    if not trace.is_file() or not uart.is_file():
        raise RuntimeError(f"simulator logs missing for {name}")
    shutil.copy2(trace, result_dir / "run.out")
    shutil.copy2(uart, result_dir / "uart.log")
    return result_dir / "run.out", result_dir / "uart.log"


def memverify(manifest: Path, trace: Path, base_addr: int | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SDK_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [sys.executable, "-m", "memverify.cute_memverify",
               "--manifest", str(manifest), "--trace", str(trace), "--tensor", "D"]
    if base_addr is not None:
        command += ["--base-addr", hex(base_addr)]
    run(command, cwd=SDK_ROOT, env=env)


def single_result_dir(args) -> Path:
    k = args.k if args.k else "native"
    return (CASE_DIR / "generated" / "results" /
            f"fpe_{args.fpe_bytes}B" / f"bias_{args.bias}" /
            f"m{args.m}_n{args.n}_k{k}_t{args.transpose}" /
            f"dtype_{args.dtype}_seed_{args.seed}")


def run_single(args) -> None:
    result_dir = single_result_dir(args)
    result_dir.mkdir(parents=True, exist_ok=True)
    _, manifest = prepare_single_case(args, result_dir)
    elf = configure_and_build("sv_fpe_datatype_mm", args)
    trace, _ = simulator_trace(args.config, elf, result_dir)
    memverify(manifest, trace)
    print(f"PASS: SDK SV FPE GEMM dtype={args.dtype} results={result_dir}")


def prepare_batch_cases(args, result_dir: Path) -> dict[int, Path]:
    specs = {
        0: (0, "zero", 64, 64, 0),
        1: (9, "random", 64, 64, 0),
        2: (10, "repeat", 64, 64, 0),
        3: (16, "zero", 64, 64, 0),
        4: (13, "random", 64, 64, 0),
        5: (1, "repeat", 64, 64, 0),
        6: (19, "random", 64, 64, 0),
        7: (0, "random", 128, 64, 1),
    }
    manifests = {}
    for case_id, (dtype, bias, m, n, transpose) in specs.items():
        header = GENERATED_DIR / f"batch_case_{case_id}.h"
        run(generator_args(args, header, dtype=dtype, bias=bias,
                           m=m, n=n, transpose=transpose))
        case_dir = result_dir / f"case_{case_id}"
        case_dir.mkdir(parents=True, exist_ok=True)
        saved_header = case_dir / f"batch_case_{case_id}.h"
        shutil.copy2(header, saved_header)
        golden_dir = case_dir / "golden"
        run([sys.executable, str(IMPORT_HEADER), "--header", str(saved_header),
             "--output", str(golden_dir)], cwd=SDK_ROOT)
        manifest = golden_dir / "manifest.json"
        normalize_manifest(manifest, saved_header)
        manifests[case_id] = manifest
    return manifests


def parse_batch_ranges(uart: Path) -> dict[int, int]:
    ranges = {}
    pattern = re.compile(r"BATCH_D_RANGE case=(\d+) base=([0-9a-fA-F]+) bytes=(\d+)")
    for line in uart.read_text().splitlines():
        match = pattern.search(line)
        if match:
            ranges[int(match.group(1))] = int(match.group(2), 16)
    return ranges


def run_batch(args) -> None:
    order = BATCH_ORDERS[args.batch_mode]
    result_dir = (CASE_DIR / "generated" / "results" /
                  f"fpe_{args.fpe_bytes}B" /
                  f"batch_mode_{args.batch_mode}_seed_{args.seed}")
    result_dir.mkdir(parents=True, exist_ok=True)
    manifests = prepare_batch_cases(args, result_dir)
    elf = configure_and_build("sv_fpe_datatype_mm_batch", args)
    trace, uart = simulator_trace(args.config, elf, result_dir)
    ranges = parse_batch_ranges(uart)
    if set(ranges) != set(range(len(order))):
        raise RuntimeError(f"unexpected batch D ranges: {ranges}")
    for slot, case_id in enumerate(order):
        memverify(manifests[case_id], trace, ranges[slot])
    print(f"PASS: SDK SV FPE batch mode={args.batch_mode} results={result_dir}")


def run_extended(args) -> None:
    """Reproduce the former broad regression without depending on cutetest."""
    scenarios = []
    for config, fpe_bytes in CONFIG_WIDTHS.items():
        for bias in ("random", "repeat"):
            scenarios.extend(
                (config, fpe_bytes, dtype, bias, 64, 64, None, 0,
                 f"bias_{bias}")
                for dtype in range(20)
            )
        scenarios.extend(
            (config, fpe_bytes, dtype, "random", 128, 128, None, 0,
             "multitile_128")
            for dtype in range(20)
        )
        scenarios.extend(
            (config, fpe_bytes, dtype, "random", 256, 256, None, 0,
             "multitile_256")
            for dtype in range(20)
        )
        scenarios.extend(
            (config, fpe_bytes, dtype, "random", 64, 64, 512, 0,
             "large_k_512")
            for dtype in range(20)
        )
        scenarios.extend(
            (config, fpe_bytes, dtype, "repeat", 128, 64, 256, 1,
             "transpose_repeat")
            for dtype in range(20)
        )

    original = vars(args).copy()
    try:
        for index, (config, fpe_bytes, dtype, bias, m, n, k, transpose, label) in \
                enumerate(scenarios, start=1):
            args.config = config
            args.fpe_bytes = fpe_bytes
            args.dtype = dtype
            args.bias = bias
            args.m = m
            args.n = n
            args.k = k
            args.transpose = transpose
            print(f"START {index}/{len(scenarios)} label={label} "
                  f"config={config} dtype={dtype}")
            run_single(args)

        for config, fpe_bytes in CONFIG_WIDTHS.items():
            args.config = config
            args.fpe_bytes = fpe_bytes
            args.batch_mode = 0
            print(f"START batch config={config}")
            run_batch(args)
    finally:
        vars(args).update(original)
    print(f"PASS: SDK SV FPE extended regression cases={len(scenarios) + len(CONFIG_WIDTHS)}")


def run_extract_pe(args) -> None:
    result_dir = CASE_DIR / "generated" / "case1_pe_vectors"
    result_dir.mkdir(parents=True, exist_ok=True)
    scratch = CASE_DIR / "generated" / "batch_case_1.h"
    run(generator_args(args, scratch, dtype=9, bias="random", m=64, n=64, transpose=0))
    binary = HOST_DIR / "extract_sv_fpe_batch_case1_pe"
    cmodel_include = CUTE_ROOT / "cute-fpe" / "cmodel" / "necessary_source"
    run(["c++", "-x", "c++", "-std=c++11", "-O2", "-DNVIDIA_ON=0",
         "-DFMAC_DUMP_OUTPUT=0", f"-I{cmodel_include}",
         str(CASE_DIR / "extract_batch_case1_pe.cpp"), "-o", str(binary)])
    run([str(binary), str(result_dir)])
    print(f"PE vectors: {result_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="CUTE4TopsSCP64CMLDebugConfig",
                        choices=sorted(CONFIG_WIDTHS))
    parser.add_argument("--fpe-bytes", type=int, choices=(32, 64))
    parser.add_argument("--dtype", type=int, default=0, choices=range(20))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bias", choices=("zero", "random", "repeat"), default="zero")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int)
    parser.add_argument("--transpose", type=int, choices=(0, 1), default=0)
    parser.add_argument("--tensor-k-bytes", type=int, default=64)
    parser.add_argument("--scale-tile-m", type=int, default=64)
    parser.add_argument("--scale-tile-n", type=int, default=64)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--all-dtypes", action="store_true",
                        help="run the selected single-GEMM parameters for CUTE opcodes 0 through 19")
    parser.add_argument("--extended", action="store_true",
                        help="run the complete 242-case multi-width GEMM regression")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch-mode", type=int, choices=sorted(BATCH_ORDERS), default=0)
    parser.add_argument("--extract-pe", action="store_true")
    args = parser.parse_args()
    if args.fpe_bytes is None:
        args.fpe_bytes = CONFIG_WIDTHS[args.config]
    if args.fpe_bytes != CONFIG_WIDTHS[args.config]:
        parser.error("--fpe-bytes must match the selected hardware CONFIG")
    selected_modes = sum((args.all_dtypes, args.extended, args.batch, args.extract_pe))
    if selected_modes > 1:
        parser.error("--all-dtypes, --extended, --batch, and --extract-pe are mutually exclusive")
    return args


def main():
    args = parse_args()
    ensure_generator()
    if args.extract_pe:
        run_extract_pe(args)
    elif args.batch:
        run_batch(args)
    elif args.extended:
        run_extended(args)
    elif args.all_dtypes:
        for dtype in range(20):
            args.dtype = dtype
            run_single(args)
        print("PASS: SDK SV FPE GEMM all dtypes 0-19")
    else:
        run_single(args)


if __name__ == "__main__":
    main()
