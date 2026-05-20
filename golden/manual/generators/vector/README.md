# Phase C0 RVV Golden Generator

用 NVWA 的 `__gloden_*` RVV reference 函数，在 QEMU 上执行后生成 `.h` 文件，
供 primitive 测试直接 `#include` 使用，做 bit-exact 比较。

## 生成流程

```
cases.yaml (配置 case + 参数)
    ↓ build.py 读取
gen_*.c (xorshift32 生成输入 + __gloden_* 计算golden)
    ↓ GCC 15.1 交叉编译 (-DGOLDEN_M=64 -DGOLDEN_N=64 ...)
gen_*.riscv (static ELF)
    ↓ QEMU 执行 (rv64,v=true,vlen=512,zvfh=true)
stdout → .h 文件 (hex float 数组)
```

## 用法

```bash
cd /root/opencute/CUTE/cute-sdk

# 生成单个 case
python3 golden/manual/generators/vector/build.py --case silu

# 生成全部
python3 golden/manual/generators/vector/build.py --all
```

输出目录：`golden/manual/vector/golden_*.h`

## 配置

所有 case 的参数（尺寸、位置等）在 `cases.yaml` 中配置：

```yaml
cases:
  - name: silu
    source: gen_silu.c
    params:
      GOLDEN_M: 64
      GOLDEN_N: 64

  - name: smoothquant
    source: gen_smoothquant.c
    params:
      GOLDEN_M: 128
      GOLDEN_K: 2048
```

修改 `params` 中的值后重新 `build.py --all` 即可生成不同尺寸的 golden。
C 源码中的尺寸宏（`GOLDEN_M`、`GOLDEN_N` 等）通过 `-D` 传入，不硬编码。

工具链也可在 `cases.yaml` 的 `toolchain` 段修改。

## 已生成的 Case

| Case 名 | 算子 | 参数 | 输出类型 |
|---|---|---|---|
| silu | SiLU | M=64, N=64 | F32 |
| vec_math | exp/sin/cos | N=256 | F32 |
| dequant_f32 | dequant I32→F32 | M=64, N=64 | F32 |
| dequant_f16 | dequant I32→F16 | M=64, N=64 | F16 (uint16) |
| resadd | element-wise add | M=64, N=64 | F32 |
| hadamard | element-wise mul + row_absmax | M=64, N=128 | F32 |
| rope_bf16_pos0 | RoPE position=0 | M=64, HEAD_DIM=64 | BF16 (uint16) |
| rope_bf16_pos17 | RoPE position=17 | M=64, HEAD_DIM=64 | BF16 (uint16) |
| masked_softmax_bf16 | causal softmax | M=64, N=128 | BF16 (uint16) |
| smoothquant | SmoothQuant F32→I8 | M=128, K=2048 | I8 + scale F32 |
| rmsnorm | RMSNorm | BATCH=1, SEQ=128, DIM=2048 | F32 |
| rmsnorm_scale | RMSNorm + per_token_scale | BATCH=1, SEQ=128, DIM=2048 | F32 + scale F32 |

## 生成的 .h 文件格式

以 silu 为例：

```c
#ifndef GOLDEN_SILU_M64_N64_F32_H
#define GOLDEN_SILU_M64_N64_F32_H

#include <stdint.h>

#define GOLDEN_SILU_M 64
#define GOLDEN_SILU_N 64
#define GOLDEN_SILU_TOTAL 4096

static const float golden_silu_input_x[4096] = {
    -0x1.ep+4, -0x1.143914p+2, ...
};

static const float golden_silu_golden_y[4096] = {
    -0x1.1e77c6p-29, 0x1.cd7224p+4, ...
};

#endif
```

F16 输出用 `uint16_t` 数组存储原始 bit，I8 输出用 `int8_t` 数组。
所有 F32 值用 `%a` hex float 格式输出，保证 bit-exact round-trip。

## 测试中使用

```c
#include "golden_silu_m64_n64.h"

float x = golden_silu_input_x[0];
float y = golden_silu_golden_y[0];
```

## 文件结构

```
golden/manual/generators/vector/
├── build.py                    # 编译+运行+生成 .h 的编排脚本
├── cases.yaml                  # case 配置（参数、工具链）
├── include/
│   ├── nvwa_gloden_opt.h       # NVWA __gloden_* 函数 (C0 清理版)
│   ├── nvwa_llama_primitives.h # 从 llama3_1B_cpu.c 拆出的 dequant/resadd/hadamard
│   └── cute_golden_inputs.h    # xorshift32 PRNG + 各算子输入生成
└── cases/
    ├── gen_silu.c
    ├── gen_vec_math.c
    ├── gen_dequant.c
    ├── gen_resadd.c
    ├── gen_hadamard.c
    ├── gen_rope.c
    ├── gen_masked_softmax.c
    ├── gen_smoothquant.c
    └── gen_rmsnorm.c
```

## 确定性

所有输入由固定种子的 xorshift32 生成（seed=0x43555445 "CUTE"），
不依赖 libc `rand()`。相同代码+相同种子+相同 QEMU 配置（VLEN=512），
生成的 golden 完全一致。
