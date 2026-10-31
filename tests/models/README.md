# cutetest BERT / LLaMA migration

These stage programs use the SDK runtime and trace-visible RVV primitives. The
same `model.c` is compiled for QEMU system mode and Verilator. No tensor output is
read back by the CPU. Each completed software task compares **all** declared
outputs, including raw accumulators, intermediate floats, and token scales.

| Cases | cutetest sources | Computation |
|---|---|---|
| bert_1 | ibert_1 / ibert-base-1 | QKV INT8 GEMM, saturated INT8, V transpose |
| bert_2 | *_2 | all heads QKᵀ, integer softmax, explicit exp workspace |
| bert_3 | *_3 | all heads probability × V, strided concatenation |
| bert_4 | *_4 | O projection, integer LayerNorm then residual |
| bert_5 | *_5 | FF1 + row bias + integer GELU |
| bert_6 | *_6 | FF2 + row bias + integer LayerNorm then residual |
| llama_1 | llama3_1B_1 / _8 | attention and FFN RMSNorm, scale and INT8 |
| llama_2 | _2 / _3 / _4 | Q/K/V projections, dequant, FP16 RoPE / V transpose |
| llama_3 | _7 | context quantization, O projection, FP32 residual |
| llama_4 | _9 / _10 / _11 | gate + SiLU, up + product, quant, down + residual |

## Build and run

From the OpenCUTE root, activate the compiler/Verilator/CMake/QEMU dependency
environment used by the S1 build. Python requires NumPy. Build directories must
be absolute, without whitespace, and outside the source tree.

```sh
python3 tools/runner/cute-model-build.py \
  --base-build-manifest /absolute/s1/build-manifest.json \
  --out /absolute/models/quick --profile quick \
  --cc /absolute/bin/riscv-none-elf-gcc --jobs 16
python3 tools/runner/cute-model-run.py \
  --build-manifest /absolute/models/quick/build-manifest.json \
  --out /absolute/new-run --jobs 4
```

Use `--cases bert_1 llama_2` to select stages. `--ack-delay 3 --marker-delay 17`
exercises completion delays. Each run directory is new. `summary.json` and the
individual `report.json` must all pass; online/offline reports must match.

| Profile | S | H | F | Q/BERT heads | KV heads |
|---|---:|---:|---:|---:|---:|
| quick | 64 | 128 | 256 | 2 | 1 |
| bert-small | 128 | 512 | 1024 | 4 | — |
| bert-base | 128 | 768 | 3072 | 12 | — |
| llama-1b | 128 | 2048 | 8192 | 32 | 8 |

Profiles use separate output/build directories. The original dimensions are
preserved; inputs are deterministic nonzero fixtures, not trained checkpoints.
This is stage validation, not an accuracy/perplexity benchmark or full decoder.
L5/M2 floating CUTE attention is excluded.

## Numerical contract and deliberate changes

- B stores N×K. GEMM issues 64×64 tiles, waits in FIFO order, and preserves
  32-bit wrapping accumulation. INT8 post-ops clamp to [-128,127]. `frm=RNE`.
- BERT softmax uses the old scale .5 integer approximation (including RVV
  shift-count masking), not floating softmax. LayerNorm uses wrapped integer
  moments, truncating mean/variance division and integer sqrt, then RNE/clamp
  **before** residual addition. GELU uses the old scale=1 constants -6/-2/2/-6.
- The old scale helper passes `dim_j` where its callee expects the row count.
  New kernels use explicit element/row counts. Hardcoded width/stack constants
  and zero-filled experiments are not copied as a portable API.
- Old LLaMA names containing BF16 actually use FP16 in these stages. The migrated
  outputs are FP16, preserving the old theta table and adjacent real/imag inputs
  with split-half RoPE outputs. This does not change existing SDK BF16 APIs.
- RMSNorm uses 64 FP32 lane sums followed by an ordered FP32 reduction. This
  fixes the sum order for this VLEN=512 profile; it is not a promise of matching
  the old implementation's unspecified unordered reduction for every input.
- SiLU uses the SDK exponential polynomial and reciprocal estimate with two
  Newton corrections, avoiding one iterative vector divide per element. This
  is an explicit numeric implementation change from the old exact divide.
  QEMU/reference calibration is recorded per output, and RTL/QEMU is byte exact.
- `cute_model_math.h` is generated from `cute_vec_math.h` with LMUL=1. The old
  LMUL=4 expression spilled vectors to the stack under GCC 15.2. The checker
  rejects such undeclared writes. Regenerate with `generate_math.py`; use
  `generate_math.py --check` to detect drift.
- Scale outputs are RVV stores, so the same ack memory image can observe them.
  No general scalar-store observer is implied. All arrays are 64-byte aligned;
  sub-cacheline vector masks (e.g. one FP32 scale) are still checked.

The NumPy oracle uses exact INT8/INT32 rules. Float mathematical references use
explicit per-output tolerance for QEMU calibration (FP32 rtol=3e-5, atol=3e-6;
FP16 rtol=.002, atol=.0001), never for RTL comparison. Every oracle/input file,
case and generated program is hashed in `fixture.json` and bound to the build.

The current scheduling path serializes tiles and drains the hart at checkpoints.
It does not preserve old TCM/double-buffer optimizations or claim performance
parity. Task/output boundaries make later pipelining changes reviewable.

L4 uses compositional calibration: after an approximate activation passes its
mathematical tolerance, later reference operations use that validated upstream
value. Hadamard and quantization are independently recomputed with NumPy, followed
by exact INT8 GEMM. This handles rounding-threshold crossings without tolerating
integer mismatches. `oracle_expression` names only already calibrated task outputs
or hashed fixture inputs; current/future outputs are rejected. Derived references
and their hashes are saved under each run's `calibration/` directory.
