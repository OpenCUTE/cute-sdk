# Phase C1: fuse_vector 实现计划

## Context

`fuse_vector` 是纯向量算子组合层。它只接收普通 buffer、stride、shape、scale 和显式 ctx，不接 `cute_post_call_t`，不管理 matmul tile 调度。

这一层应该归到：

- 实现：`cutelib/primitive`
- 测试：`tests/primitive`
- golden：`golden/manual/vector`

这样可以先把 dequant + SiLU、dequant + residual add、dequant + RoPE + BF16 等纯 vector fuse 算子单独验证清楚，再交给 tensor fusion 层复用。

---

## 1. 范围

| 原函数 | 新 primitive API | primitive case |
|---|---|---|
| `fuse_ops_DEQUANT_ROPE_BF16CVRT` | `cute_fuse_dequant_rope_bf16cvt_tile` | `primitive_fuse_dequant_rope_bf16cvt_m64_n64` |
| `fuse_ops_DEQUANT_BF16CVRT_With_T` | `cute_fuse_dequant_bf16cvt_tile` | `primitive_fuse_dequant_bf16cvt_m64_n64` |
| `fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT` | `cute_fuse_masked_softmax_kvscale_bf16cvt_tile` | `primitive_fuse_masked_softmax_kvscale_bf16cvt_m64_n128` |
| `fuse_ops_DEQUANT_RESADD` | `cute_fuse_dequant_resadd_tile` | `primitive_fuse_dequant_resadd_m64_n64` |
| `fuse_ops_DEQUANT_SILU` | `cute_fuse_dequant_silu_tile` | `primitive_fuse_dequant_silu_m64_n64` |
| `fuse_ops_DEQUANT_HADAMARD` | `cute_fuse_dequant_hadamard_tile` | `primitive_fuse_dequant_hadamard_m64_n128` |

不属于本计划：

- `cute_post_call_t` adapter。
- `cute_tiled_matmul` notile / nopipeline / pipeline 调度。
- layer/model buffer 生命周期管理。

---

## 2. 文件结构

```text
cutelib/primitive/include/
    cute_vector_fusion.h

tests/primitive/
    primitive_fuse_dequant_rope_bf16cvt_m64_n64/
        case.json
        test.c
    primitive_fuse_dequant_bf16cvt_m64_n64/
        case.json
        test.c
    primitive_fuse_masked_softmax_kvscale_bf16cvt_m64_n128/
        case.json
        test.c
    primitive_fuse_dequant_silu_m64_n64/
        case.json
        test.c
    primitive_fuse_dequant_hadamard_m64_n128/
        case.json
        test.c
    primitive_fuse_dequant_resadd_m64_n64/
        case.json
        test.c

golden/manual/vector/
    fuse_dequant_rope_bf16cvt_m64_n64/
    fuse_dequant_bf16cvt_m64_n64/
    fuse_masked_softmax_kvscale_bf16cvt_m64_n128/
    fuse_dequant_silu_m64_n64/
    fuse_dequant_hadamard_m64_n128/
    fuse_dequant_resadd_m64_n64/
```

每个 primitive case 仍然是一目录一个 `case.json`、一个 `test.c`、一个 binary。

---

## 3. API

`cutelib/primitive/include/cute_vector_fusion.h`：

```c
#ifndef CUTE_VECTOR_FUSION_H
#define CUTE_VECTOR_FUSION_H

#include <stdint.h>
#include <stddef.h>
#include "cute_convert.h"
#include "cute_elementwise.h"
#include "cute_sequence.h"

typedef struct {
    int pos;
    const float *rope_theta;
    int key_dim;
} cute_rope_ctx_t;

typedef struct {
    int pos;
    const int8_t *bitmask;
    int max_ctx_len;
    float kv_scale;
} cute_softmax_ctx_t;

typedef struct {
    float *residual;
    uint64_t residual_stride;
} cute_resadd_ctx_t;

typedef struct {
    const float *lhs;
    uint64_t lhs_stride;
    float *output_absmax;
} cute_hadamard_ctx_t;

void cute_fuse_dequant_rope_bf16cvt_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols,
    const cute_rope_ctx_t *ctx);

void cute_fuse_dequant_bf16cvt_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols);

void cute_fuse_masked_softmax_kvscale_bf16cvt_tile(
    const float *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    int row0, int col0,
    int rows, int cols,
    const cute_softmax_ctx_t *ctx);

void cute_fuse_dequant_silu_tile(
    const int32_t *input, uint64_t input_stride,
    float *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols);

void cute_fuse_dequant_hadamard_tile(
    const int32_t *input, uint64_t input_stride,
    float *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols,
    const cute_hadamard_ctx_t *ctx);

void cute_fuse_dequant_resadd_tile(
    const int32_t *input, uint64_t input_stride,
    float *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols,
    const cute_resadd_ctx_t *ctx);

#endif /* CUTE_VECTOR_FUSION_H */
```

命名使用 `_tile`，表示处理一个显式 shape/stride 的 vector tile。它不是 tensor post-op，也不是调度入口。

---

## 4. 实现约定

每个 fuse primitive 只组合 Phase C0 已验证的 primitive。例如 dequant + SiLU：

```c
cute_dequant_i32_to_f32_tile(input, input_stride,
                             output, output_stride,
                             input_scale, weight_scale,
                             rows, cols);
cute_silu_tile(output, output_stride, rows, cols);
```

要求：

- 不读取 `cute_post_call_t`。
- 不读取 `llama3_1B.c` 全局变量。
- 不引入新的近似数学路径作为默认实现。
- `input_scale` 必须已经指向当前 row 起点。
- BF16 转换、RoPE、softmax 的数学顺序与原实现一致。
- Hadamard 的 `output_absmax[row]` 在多 N tile 下要做 max 累积。

---

## 5. case.json

示例：`tests/primitive/primitive_fuse_dequant_resadd_m64_n64/case.json`

```json
{
  "id": "primitive_fuse_dequant_resadd_m64_n64",
  "op_ref": "ops/vector/resadd.yaml",
  "level": "primitive",
  "build": {
    "source": "test.c",
    "target": "test.riscv"
  },
  "run": {
    "hwconfig": "cute4tops_shuttle512_d512_v512_m512_sysbus512_membus1_core_dramsim48",
    "trace_source": "run.out"
  },
  "golden": "golden/manual/vector/fuse_dequant_resadd_m64_n64/manifest.json",
  "verify": {
    "mode": "return_code_and_bit_exact",
    "tensors": [
      {
        "tensor": "golden_output",
        "symbol": "output"
      }
    ]
  }
}
```

这里不需要扩展现有 manifest schema。

---

## 6. 测试矩阵

| case | 验证内容 |
|---|---|
| `primitive_fuse_dequant_silu_m64_n64` | dequant + SiLU bit-exact |
| `primitive_fuse_dequant_resadd_m64_n64` | dequant + residual add bit-exact |
| `primitive_fuse_dequant_bf16cvt_m64_n64` | dequant + BF16 convert + layout bit-exact |
| `primitive_fuse_dequant_rope_bf16cvt_m64_n64` | dequant + RoPE + BF16 bit-exact |
| `primitive_fuse_dequant_hadamard_m64_n128` | hadamard + absmax 跨 N tile 累积 |
| `primitive_fuse_masked_softmax_kvscale_bf16cvt_m64_n128` | masked softmax + kv scale + BF16，允许 0.1% 浮点误差 |

建议新增 `tests/vecfusion.yaml`，避免把普通 primitive 和 fuse primitive 混在一个 suite 里。

---

## 7. 执行步骤

| 步骤 | 内容 |
|---|---|
| 1 | 创建 `cute_vector_fusion.h` |
| 2 | 实现 SiLU / ResAdd / BF16 三个简单 fuse primitive |
| 3 | 添加对应 golden 和 `tests/primitive/primitive_fuse_*` |
| 4 | 实现 RoPE / Hadamard / Masked Softmax 三个复杂 fuse primitive |
| 5 | 添加剩余 golden 和 `tests/primitive/primitive_fuse_*` |
| 6 | 更新 CMake 和 `tests/vecfusion.yaml` |
| 7 | 跑完整 fuse_vector suite |

验证命令：

```bash
cd /root/opencute/CUTE
cmake --build build -j$(nproc)
python3 tools/runner/cute-test.py --suite cute-sdk/tests/vecfusion.yaml --skip-build
```

---

## 8. Done Criteria

1. 6 个 `cute_fuse_*_tile` API 都可独立调用。
2. 6 个 `primitive_fuse_*` case 都有固定 golden。
3. 整数/定点/简单 F32 case bit-exact 通过；softmax/exp/归约类 case 允许明确标注的浮点误差。
4. 所有函数不依赖 `cute_post_call_t`。
5. 所有函数不依赖 `llama3_1B.c` 全局变量。
