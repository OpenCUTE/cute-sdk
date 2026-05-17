# Phase C0 RVV golden generation design

## Goal

Phase C0 的 golden 不再用 Python/NumPy 作为最终真值源，而是用
RVV reference binary 在 `cuteqemu/build/qemu-riscv64` 上执行后 dump：

```text
golden/manual/vector/<case_id>/
    manifest.json
    input_*.bin
    golden*.bin
```

这样 `vec_exp`、`vec_sin`、`vec_cos`、F32->F16/BF16 narrowing、INT8
rounding、softmax reduction 这些路径都来自同一套 RVV 指令行为，后续
primitive 测试可以做 bit-exact。

---

## Source of truth

优先使用 NVWA 里的 RVV reference，不直接依赖 x86 fallback。

```text
/root/opencute/opencute_github/NVWA/
    llama3.2_1B/data_flow/gloden_opt.h
    rmsnorm/intrinsic.c
    silu/intrinsic.c
    rope/intrinsic.c
    softmax/intrinsic.c
    smoothquant01/intrinsic.c
    cvrtfp16/intrinsic.c
```

推荐以 `gloden_opt.h` 为第一来源，因为它已经把很多符号做成
`__gloden_*` 命名空间：

```c
__gloden_vec_exp
__gloden_vec_sin
__gloden_vec_cos
__gloden_RMSnorm
__gloden_rope
__gloden_silu
__gloden_smoothquantO1
__gloden_softmax
__gloden_cvrtfp16
```

`hadamard`、`resadd`、`dequant` 在 NVWA standalone op 里不是完整单算子，
应从 `llama3.2_1B/data_flow/llama3_1B_cpu.c` 的 fusion 函数中拆出 stage：

```text
fuse_ops_DEQUANT_RESADD       -> dequant_i32_to_f32 + resadd
fuse_ops_DEQUANT_SILU         -> dequant_i32_to_f32 + silu
fuse_ops_DEQUANT_HADAMARD     -> dequant_i32_to_f32 + hadamard
fuse_ops_DEQUANT_BF16CVRT     -> dequant_i32_to_f32 + cvt_to_storage
```

拆分原则：只删 fusion glue，不改常量、不改 intrinsic 顺序、不改 rounding
路径。

---

## Directory layout

建议新增一个专门的 manual golden generator 区，不塞进 `tests/primitive`：

```text
golden/manual/generators/vector/
    README.md
    include/
        cute_golden_dump.h          # write_all / manifest helper
        cute_golden_inputs.h        # deterministic input generators
        cute_golden_manifest.h      # JSON string emitter
        nvwa_gloden_opt.h           # copied/adapted from NVWA gloden_opt.h
        nvwa_llama_primitives.h     # dequant/resadd/hadamard拆分函数
    cases/
        gen_vec_math.c
        gen_dequant.c
        gen_silu.c
        gen_hadamard.c
        gen_resadd.c
        gen_rope.c
        gen_masked_softmax.c
        gen_smoothquant.c
        gen_rmsnorm.c
    build.py                       # compile + run all/single case
```

输出目录继续沿用现有 golden 习惯：

```text
golden/manual/vector/
    vec_math_basic/
    silu_m64_n64_f32/
    rope_m64_head64_pos0_f32_to_f16/
    rope_m64_head64_pos17_f32_to_f16/
    masked_softmax_m64_n128_causal_f16/
    smoothquant_m128_k2048/
    rmsnorm_b1_s128_h2048/
```

---

## Build and run flow

Host 侧只负责编译 RISC-V static binary，然后用 QEMU 跑。第一版不需要
CMake，避免污染 SDK 主构建。

```bash
cd /root/opencute/CUTE/cute-sdk

python3 golden/manual/generators/vector/build.py --case silu
python3 golden/manual/generators/vector/build.py --all
```

`build.py` 执行的底层命令：

```bash
riscv64-linux-gnu-gcc \
  -O2 \
  -static \
  -march=rv64gcv \
  -mabi=lp64d \
  -I golden/manual/generators/vector/include \
  golden/manual/generators/vector/cases/gen_silu.c \
  -lm \
  -o build/golden/manual/vector/gen_silu.riscv

cuteqemu/build/qemu-riscv64 \
  -cpu rv64,v=true \
  build/golden/manual/vector/gen_silu.riscv \
  golden/manual/vector/silu_m64_n64_f32
```

每个 generator binary 接受一个参数：输出目录。binary 内部创建目录、写
`.bin`、写 `manifest.json`。

---

## Binary format

所有 `.bin` 直接写 little-endian raw tensor bytes，和现有 tensor golden
一致。

| dtype | element_bits | bin encoding |
|---|---:|---|
| `F32` | 32 | IEEE754 little-endian float |
| `I32` | 32 | little-endian signed int32 |
| `I8` | 8 | signed int8 |
| `U8` | 8 | unsigned byte |
| `F16` | 16 | `_Float16` RVV `vfncvt.f.f.w` storage bits |
| `BF16` | 16 | BF16 storage bits, only after BF16 path is explicitly implemented |
| `U1_PACKED` | 1 | bit-packed mask, bit `i % 8` in byte `i / 8` |

注意：当前 NVWA `cvrtfp16` 和 llama fusion 代码实际使用的是 `_Float16`
路径：

```c
vfloat16m2_t y = __riscv_vfncvt_f_f_w_f16m2(x, vl);
__riscv_vse16_v_f16m2((_Float16*)dst, y, vl);
```

所以 manifest 第一版必须写 `F16`，不要笼统写 `BF16_OR_F16_STORAGE`。等
真正接入 BF16 intrinsic，如 `vfncvtbf16.f.f.w` 或 scalar `fcvt.bf16.s`，
再新增 `*_bf16` case。

---

## Manifest schema

沿用现有 `manifest.json` 的顶层结构，但扩展 dtype 和 shape 维度：

```json
{
  "id": "silu_m64_n64_f32",
  "op": "silu",
  "level": "primitive",
  "op_ref": "ops/vector/silu.yaml",
  "tensors": {
    "X": {
      "path": "input_x.bin",
      "dtype": "F32",
      "element_bits": 32,
      "layout": "row_major",
      "shape": [64, 64],
      "stride_bytes": 256,
      "total_bytes": 16384
    },
    "Y": {
      "path": "golden_y.bin",
      "dtype": "F32",
      "element_bits": 32,
      "layout": "row_major",
      "shape": [64, 64],
      "stride_bytes": 256,
      "total_bytes": 16384
    }
  },
  "attributes": {
    "M": 64,
    "N": 64
  },
  "verify": {
    "mode": "bit_exact",
    "tensors": ["Y"]
  },
  "generator": {
    "tool": "rvv_qemu",
    "source": "NVWA/llama3.2_1B/data_flow/gloden_opt.h",
    "qemu": "cuteqemu/build/qemu-riscv64",
    "qemu_cpu": "rv64,v=true",
    "compiler": "riscv64-linux-gnu-gcc",
    "cflags": "-O2 -static -march=rv64gcv -mabi=lp64d",
    "created": "YYYY-MM-DD"
  }
}
```

`GoldenTensor` reader 当前只完整支持 2D integer tensor。Phase C0 测试前需要
升级 reader：

1. 支持 `F32/F16/BF16/U8/U1_PACKED/I32/I8`。
2. 支持 `shape` 为 1D、2D、3D，但访问验证可以先只用 raw bytes。
3. 当 `total_bytes` 存在时优先使用它。

---

## Deterministic input generation

不要依赖 libc `rand()`，避免 host/libc 差异。generator 里用固定的
`xorshift32` 或 LCG 生成输入，并把输入也 dump 到 `.bin`。

推荐模式：

```c
static uint32_t rng = 0x43555445u; /* CUTE */

static uint32_t xorshift32(void) {
    uint32_t x = rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng = x;
    return x;
}
```

F32 输入不要只用 `[0, 1]`，每个 op 需要覆盖关键区间：

| op | input pattern |
|---|---|
| vec_exp | `[-90, -20, -1, 0, 1, 20, 80]` + random |
| vec_sin/cos | `[-8pi, -pi, -pi/2, 0, pi/2, pi, 8pi]` + random |
| silu | `[-30, 30]`，覆盖 exp overflow/underflow 附近 |
| rope | normal range F32，`rope_theta[k] = powf(10000, -2k/head_dim)` |
| softmax | row-wise mixed values，mask 覆盖 full/causal/sparse |
| smoothquant | rows 含正负极值和接近 0 的值 |
| rmsnorm | normal range，含小值行和大值行 |
| dequant | I32 覆盖正负累加值，scale 覆盖非 2 的幂 |

---

## Case matrix

第一批 golden 只做小而稳的 case，够验证单 primitive。

| case id | op | shape | output |
|---|---|---:|---|
| `vec_math_exp_sin_cos_n256` | vec_math | `[256]` | `exp/sin/cos` F32 |
| `dequant_m64_n64_i32_to_f32` | dequant | `[64,64]` | F32 |
| `dequant_m64_n64_i32_to_f16` | dequant | `[64,64]` | F16 storage |
| `silu_m64_n64_f32` | silu | `[64,64]` | F32 |
| `resadd_m64_n64_f32` | resadd | `[64,64]` | F32 |
| `hadamard_m64_n128_f32` | hadamard | `[64,128]` | F32 + row_absmax |
| `rope_m64_head64_pos0_f16` | rope | `[64,64]` | F16 storage |
| `rope_m64_head64_pos17_f16` | rope | `[64,64]` | F16 storage |
| `masked_softmax_m64_n128_causal_f16` | masked_softmax | `[64,128]` | F16 storage |
| `smoothquant_m128_k2048` | smoothquant | `[128,2048]` | I8 + scale F32 |
| `rmsnorm_b1_s128_h2048` | rmsnorm | `[1,128,2048]` | F32 |
| `rmsnorm_scale_b1_s128_h2048` | rmsnorm_with_scale | `[1,128,2048]` | F32 + per_token_scale |

---

## Migration details per op

### vec_math

从 `gloden_opt.h` 复制：

```c
__gloden_vec_exp
__gloden_vec_sin
__gloden_vec_cos
```

generator 加载 F32 vector，分别 dump `golden_exp.bin`、`golden_sin.bin`、
`golden_cos.bin`。

### SiLU

直接调用：

```c
__gloden_silu(input, output, 1, M, N);
```

输出 `Y` 为 F32。

### RoPE

直接调用：

```c
__gloden_rope(input, tmp_f32, rope_theta, pos, 1, 1, M, head_dim);
__gloden_cvrtfp16(tmp_f32, output_f16, M, head_dim);
```

manifest 里同时记录 `golden_y_f32.bin` 可选 debug 输出，bit-exact 验证默认
用 `golden_y.bin` 的 F16 storage。

### Masked softmax

第一版用 `__gloden_softmax` 生成 F32 softmax，再用 `__gloden_cvrtfp16`
生成 F16 storage：

```c
__gloden_softmax(input_scaled, tmp_f32, mask_packed, M, N);
__gloden_cvrtfp16(tmp_f32, output_f16, M, N);
```

如果测试目标 primitive 内部包含 `scale`，则 generator 应先 dump unscaled
`X`，再在 reference 内做 `X * scale`，避免输入语义错位。

### Smoothquant

直接调用：

```c
__gloden_smoothquantO1(input, output_i8, scale_f32, M, K);
```

需要额外覆盖 `scale[i] == 0` 风险。当前 NVWA stage2 没有 guard：

```c
float id = 1.0f / scale[i];
```

因此 golden input 不应生成全 0 row，除非我们明确要把这个行为固化成测试。

### RMSNorm

直接调用：

```c
__gloden_RMSnorm(input, output, weight, eps, batch, seq_len, hidden_dim);
```

`RMSnorm_With_getabsmax_scale` 不在 `gloden_opt.h`，从
`llama3_1B_cpu.c` 拆到 `nvwa_llama_primitives.h`，命名为：

```c
__gloden_RMSnorm_with_getabsmax_scale(...)
```

### Dequant / ResAdd / Hadamard

从 fusion stage 拆出三个纯函数：

```c
__gloden_dequant_i32_to_f32(input_i32, input_stride, output_f32,
                            output_stride, input_scale, weight_scale,
                            M, N);

__gloden_resadd_f32(lhs_f32, lhs_stride, rhs_f32, rhs_stride,
                    output_f32, output_stride, M, N);

__gloden_hadamard_f32(lhs_f32, lhs_stride, rhs_f32, rhs_stride,
                      output_f32, output_stride, row_absmax, M, N);
```

这里不再把 `dequant` 混进 `resadd/hadamard/silu` 的 C0 golden。fusion 组合留给
Phase C1。

---

## Implementation order

1. 写 `cute_golden_dump.h`：`write_file`、`mkdir_p`、`emit_manifest`。
2. 拷贝并清理 `nvwa_gloden_opt.h`：只保留 C0 需要的 `__gloden_*`。
3. 写 `gen_silu.c` 作为第一条竖切链路，确认 `.bin + manifest` 能生成。
4. 写 `build.py --case silu`，跑通 QEMU。
5. 升级 `GoldenTensor` reader 支持 F32/F16/raw bytes。
6. 按 case matrix 补齐其他 generator。
7. 每生成一个 case，就在 `tests/primitive/<case>/case.json` 引用对应 manifest。

---

## Acceptance criteria

1. 所有 vector golden 都由 RISC-V static binary 在 RVV QEMU 下生成。
2. 每个 case 的输入和输出都在 manifest 中声明，测试不再依赖隐藏全局数组。
3. F32/I32/I8/F16 storage 的 `.bin` 都可被 reader 读取或 raw-byte 比较。
4. C0 primitive 测试只验证单算子，不夹带 matmul/fusion/pipeline。
5. C1 fusion 可以复用 C0 的 primitive golden 组合出更复杂 case。

---

## Implementation details

### Manifest generation strategy

manifest.json 由 `build.py` 在 host 侧生成，C binary 只负责 dump `.bin` 文件。C binary
通过 stdout 输出 tensor 元数据（name, dtype, shape），build.py 据此组装 JSON。

### QEMU VLEN 锁定

QEMU 命令使用 `-cpu rv64,v=true,vlen=512`，确保 VLEN 不变。

### nvwa_gloden_opt.h 清理范围

从 `gloden_opt.h` 保留：
- `__gloden_vec_exp`, `__gloden_vec_sin`, `__gloden_vec_cos`, `__gloden_vec_sin_small`, `__gloden_vec_tanh`
- `__gloden_silu`, `__gloden_rope`, `__gloden_softmax`
- `__gloden_smoothquantO1`, `__gloden_smoothquantO1_stage1_getscale`, `__gloden_smoothquantO1_stage2_quant`
- `__gloden_cvrtfp16`, `__gloden_RMSnorm`
- `fast_sqrt`

移除：`__gloden_f16_matmul`, `__gloden_GeLu`, `__gloden_Q_matmul_I8I8I32`, `__gloden_pertoken_pertensor_scale`, `check_diff_*`

### nvwa_llama_primitives.h 拆分

从 `llama3_1B_cpu.c` 拆出纯函数，去掉 fusion glue：

- `__gloden_dequant_i32_to_f32(input_i32, input_scale, weight_scale, M, N)`
- `__gloden_resadd_f32(lhs, rhs, M, N)`
- `__gloden_hadamard_f32(lhs, rhs, row_absmax_out, M, N)` — 补全 NVWA 中被注释的 absmax 计算
- `__gloden_RMSnorm_with_getabsmax_scale(...)` — 来自 llama3_1B_cpu.c:1067

### GoldenTensor reader 升级要点

- 支持 F32 (`<f`), F16 (`<e`), BF16, U8, I8, U1_PACKED
- 支持 1D/2D/3D shape
- 当 `total_bytes` 存在时优先使用
