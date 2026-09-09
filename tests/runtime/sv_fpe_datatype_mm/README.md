# SV FPE Runtime GEMM

This SDK runtime case replaces the legacy `cutetest/sv_fpe_datatype_mm_test`
flow. It generates A/B/C/scale data and C-model golden values, builds a
RISC-V program through the SDK CMake target, runs the parent CUTE simulator,
and uses SDK `memverify` to compare CMemoryLoader stores with the golden.

Use the parent-level runner:

```bash
python3 tools/run_sv_fpe_gemm.py --help
```

The runner first invokes `generate_gemm_golden.cpp` on the host. That program
uses `CUTE/cute-fpe/cmodel` to generate `case_data.h`, including A/B/C, any
block-scale arrays, and bit-exact D golden values. CMake then cross-compiles
`test.c` into a RISC-V ELF. The ELF calls `cute_matmul()` or
`cute_blockscale_matmul()` from `cute_runtime.h`; after simulation, `memverify`
compares CMemoryLoader stores with the saved D golden.

Examples:

```bash
# One 512-bit, 64x64 zero-C INT8 GEMM.
python3 tools/run_sv_fpe_gemm.py \
  --config CUTE4TopsSCP64CMLDebugConfig --dtype 0 --bias zero

# One 256-bit NVFP4 GEMM with random C/bias.
python3 tools/run_sv_fpe_gemm.py \
  --config CUTE2TopsSCP64CMLDebugConfig --dtype 9 --bias random

# All currently defined CUTE opcodes, 0 through 19.
python3 tools/run_sv_fpe_gemm.py \
  --config CUTE4TopsSCP64CMLDebugConfig --all-dtypes --bias random

# Eight back-to-back macro tasks: FIFO, scale/non-scale transitions,
# independent D buffers, and one transposed case.
python3 tools/run_sv_fpe_gemm.py \
  --config CUTE4TopsSCP64CMLDebugConfig --batch --batch-mode 0

# Broad two-width regression: 242 cases covering all dtypes, random/repeat C,
# multi-tile M/N, K=512, transpose, and batch. It can take a long time.
python3 tools/run_sv_fpe_gemm.py --extended

# Export the real 256-bit NVFP4 PE transactions used by batch case 1.
python3 tools/run_sv_fpe_gemm.py \
  --config CUTE2TopsSCP64CMLDebugConfig --extract-pe
```

`--fpe-bytes` is checked against the selected configuration: 32 bytes for
`CUTE2TopsSCP64CMLDebugConfig` and 64 bytes for
`CUTE4TopsSCP64CMLDebugConfig`. For a single GEMM, `--m`, `--n`, `--k`,
`--bias`, and `--transpose` remain configurable. All reproducible artifacts
are under `generated/results/` and are deliberately not source-controlled.

The generated files are intentionally kept under `generated/` and are not
source-controlled. The test source uses `cute_runtime.h`, not the legacy
`cuteMarcoinstHelper.h`.
