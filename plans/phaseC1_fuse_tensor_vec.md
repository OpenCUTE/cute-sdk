# Phase C1: fuse_tensor_vec 实现计划

## Context

`fuse_tensor_vec` 是 tensor tile_op 与 vector fusion primitive 的组合层。它不重新实现数学核心，只把 `cute_tiled_matmul` 产出的 tile 通过 post-op adapter 交给 `fuse_vector` 中的 `cute_fuse_*_tile` API。

这一层归到：

- 实现：`cutelib/fusion`
- 测试：`tests/fusion`
- golden：`golden/manual/fusion`

测试组织方式：

- 一个 fusion case 目录对应一个 `case.json`。
- 一个 `case.json` 描述 `notile` / `nopipeline` / `pipeline` 三个 build variant。
- 三个 variant 分别生成三个 `.riscv` binary。
- 三个 binary 共用同一份 golden。

这需要对 CMake 和 runner 做小扩展，但不需要把三个模式塞进同一个 binary。

---

## 1. 前置依赖

必须先完成 [`phaseC1_fuse_vector.md`](phaseC1_fuse_vector.md)：

- `cute_fuse_dequant_rope_bf16cvt_tile`
- `cute_fuse_dequant_bf16cvt_tile`
- `cute_fuse_masked_softmax_kvscale_bf16cvt_tile`
- `cute_fuse_dequant_silu_tile`
- `cute_fuse_dequant_hadamard_tile`
- `cute_fuse_dequant_resadd_tile`

`fuse_tensor_vec` 的 adapter 只能调用这些 API，不能复制数学逻辑。

---

## 2. 范围

| Vector primitive API | Post-op adapter |
|---|---|
| `cute_fuse_dequant_rope_bf16cvt_tile` | `cute_post_dequant_rope_bf16cvt` |
| `cute_fuse_dequant_bf16cvt_tile` | `cute_post_dequant_bf16cvt` |
| `cute_fuse_masked_softmax_kvscale_bf16cvt_tile` | `cute_post_masked_softmax_kvscale_bf16cvt` |
| `cute_fuse_dequant_resadd_tile` | `cute_post_dequant_resadd` |
| `cute_fuse_dequant_silu_tile` | `cute_post_dequant_silu` |
| `cute_fuse_dequant_hadamard_tile` | `cute_post_dequant_hadamard` |

---

## 3. 文件结构

```text
cutelib/fusion/include/
    cute_fusion.h

tests/fusion/
    fusion_matmul_dequant_silu/
        case.json
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    fusion_matmul_dequant_resadd/
        case.json
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    fusion_matmul_dequant_bf16cvt/
        case.json
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    fusion_matmul_dequant_rope_bf16cvt/
        case.json
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    fusion_matmul_dequant_hadamard/
        case.json
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    fusion_matmul_masked_softmax_kvscale_bf16cvt/
        case.json
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
```

其中 `silu` / `resadd` / `bf16cvt` 是第一批闭环 case；RoPE / Hadamard / Masked Softmax 按下面细化计划继续补。

---

## 4. API

`cutelib/fusion/include/cute_fusion.h`：

```c
#ifndef CUTE_FUSION_H
#define CUTE_FUSION_H

#include "cute_tensor.h"
#include "cute_vector_fusion.h"

void cute_post_dequant_rope_bf16cvt(const cute_post_call_t *call);
void cute_post_dequant_bf16cvt(const cute_post_call_t *call);
void cute_post_masked_softmax_kvscale_bf16cvt(const cute_post_call_t *call);
void cute_post_dequant_silu(const cute_post_call_t *call);
void cute_post_dequant_hadamard(const cute_post_call_t *call);
void cute_post_dequant_resadd(const cute_post_call_t *call);

#endif /* CUTE_FUSION_H */
```

adapter 示例：

```c
void cute_post_dequant_silu(const cute_post_call_t *call)
{
    cute_fuse_dequant_silu_tile(
        (const int32_t *)call->tile.src,
        call->tile.src_stride,
        (float *)call->tile.dst,
        call->tile.dst_stride,
        call->env.a_scale,
        call->env.b_scale,
        call->tile.rows,
        call->tile.cols);
}
```

`cute_run_post_op()` 已经把 `a_scale` 偏移到当前 tile row：

```c
call.env.a_scale = a_scale ? a_scale + call.tile.row0 : NULL;
```

所以 adapter 内不能再次叠加 `row0`。

---

## 5. case.json: 一个 case，三个 binary

`tests/fusion/fusion_matmul_dequant_resadd/case.json` 建议写成 build variants：

```json
{
  "id": "fusion_matmul_dequant_resadd",
  "level": "fusion",
  "build": {
    "variants": [
      {
        "name": "notile",
        "source": "test_notile.c",
        "target": "fusion_matmul_dequant_resadd_notile.riscv"
      },
      {
        "name": "nopipeline",
        "source": "test_nopipeline.c",
        "target": "fusion_matmul_dequant_resadd_nopipeline.riscv"
      },
      {
        "name": "pipeline",
        "source": "test_pipeline.c",
        "target": "fusion_matmul_dequant_resadd_pipeline.riscv"
      }
    ]
  },
  "run": {
    "hwconfig": "cute4tops_shuttle512_d512_v512_m512_sysbus512_membus1_core_dramsim48",
    "trace_source": "run.out"
  },
  "golden": "golden/manual/fusion/matmul_dequant_resadd_m128_n128/manifest.json",
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

语义：

- `build.variants[*]` 每一项生成一个 binary。
- 三个 binary 使用同一个 `golden`。
- 三个 binary 可以使用同一个输出符号名 `output`，因为它们是不同 ELF。
- 如某个 variant 需要不同输出符号，可以在 variant 内覆盖 `verify`，但默认不需要。

---

## 6. Runner / CMake 扩展

### 6.1 CMake

新增 `cutelib_fusion`：

```cmake
add_library(cutelib_fusion INTERFACE)
target_include_directories(cutelib_fusion INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/fusion/include
)
target_link_libraries(cutelib_fusion INTERFACE cutelib_tensor cutelib_primitive)
```

新增 fusion variant helper：

```cmake
function(add_fusion_test_variant case_dir variant source)
    set(case_name ${case_dir}_${variant})
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusion/${case_dir}/${source})

    add_executable(${case_name} ${test_src})
    target_include_directories(${case_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/fusion/${case_dir}
    )
    target_link_libraries(${case_name} PRIVATE
        cutelib_fusion
        cute_riscv_vector_baremetal_options
        m
    )
    set_target_properties(${case_name} PROPERTIES
        SUFFIX .riscv
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/fusion
    )
endfunction()
```

第一版可以手写：

```cmake
add_fusion_test_variant(fusion_matmul_dequant_resadd notile test_notile.c)
add_fusion_test_variant(fusion_matmul_dequant_resadd nopipeline test_nopipeline.c)
add_fusion_test_variant(fusion_matmul_dequant_resadd pipeline test_pipeline.c)
```

后续再考虑从 `case.json` 自动生成 CMake。

### 6.2 Runner

`tools/runner/cute-test.py` 需要支持：

1. `_find_case_dir()` 查找 `tests/fusion`。
2. `_find_binary()` 查找 `build/fusion`。
3. `validate_case()` 接受 `build.source` 或 `build.variants`。
4. suite 里写一个 case id 时，自动展开其所有 variants。
5. 调试时允许指定单个 variant，例如 `fusion_matmul_dequant_resadd:nopipeline`。

展开后的逻辑等价于跑：

```text
fusion_matmul_dequant_resadd:notile
fusion_matmul_dequant_resadd:nopipeline
fusion_matmul_dequant_resadd:pipeline
```

每个 variant 单独 simulation，单独 memverify，但读取同一个 golden manifest。

---

## 7. 测试矩阵

| case | variants | 验证内容 |
|---|---|---|
| `fusion_matmul_dequant_silu` | `notile`, `nopipeline`, `pipeline` | matmul tile 输出 + dequant + SiLU |
| `fusion_matmul_dequant_resadd` | `notile`, `nopipeline`, `pipeline` | matmul tile 输出 + dequant + residual add |
| `fusion_matmul_dequant_bf16cvt` | `notile`, `nopipeline`, `pipeline` | matmul tile 输出 + dequant + BF16 layout |
| `fusion_matmul_dequant_rope_bf16cvt` | `notile`, `nopipeline`, `pipeline` | q/k projection + RoPE + BF16 |
| `fusion_matmul_dequant_hadamard` | `notile`, `nopipeline`, `pipeline` | FFN up + hadamard + absmax |
| `fusion_matmul_masked_softmax_kvscale_bf16cvt` | `notile`, `nopipeline`, `pipeline` | scores + mask + softmax + BF16 |

建议先做 `fusion_matmul_dequant_silu` 闭环，确认 runner 对 variants 的展开、仿真和 memverify 都正确，再复制到其他 post-op。

---

## 7.1 当前完成状态

| case | 状态 | 备注 |
|---|---|---|
| `fusion_matmul_dequant_silu` | done | 三 variant 通过，浮点容差验证 |
| `fusion_matmul_dequant_resadd` | done | 三 variant 通过，浮点容差验证 |
| `fusion_matmul_dequant_bf16cvt` | done | 三 variant 通过，修复过 f16/bf16 输出 tile 偏移 |
| `fusion_matmul_dequant_rope_bf16cvt` | pending | adapter 已有，case/golden 未补 |
| `fusion_matmul_dequant_hadamard` | pending | adapter 已有，case/golden 未补 |
| `fusion_matmul_masked_softmax_kvscale_bf16cvt` | pending | adapter 已有，但输入必须是 F32 scores，需要单独设计 |

---

## 7.2 剩余 case 细化

### 7.2.1 `fusion_matmul_dequant_rope_bf16cvt`

定位：

- 验证 q/k projection 类输出：`I8I8I32 matmul acc -> dequant -> RoPE -> BF16/F16 output`。
- `RoPE` 的 `head_dim` 是 64，所以第一版让 `N = 64`，保证每个 post-op tile 的列范围就是一个完整 head，不跨 tile。
- `M = 128`，有两个 tile row；`N = 64`，只有一个 tile col。这样仍能覆盖 `row0` 偏移，同时避免 RoPE head 被切开。

建议 shape：

| 参数 | 值 |
|---|---|
| M | 128 |
| N | 64 |
| K | 128 |
| tile | 64x64 |
| matmul dtype | `CUTEDataTypeI8I8I32` |
| post-op input | int32 acc |
| output dtype | F16/BF16 storage, manifest 先沿用 `F16` |
| output symbol | `output` |

ctx：

```c
static const cute_rope_ctx_t ctx = {
    .pos = 17,
    .rope_theta = rope_theta,
    .key_dim = 64,
};
```

实现要点：

- `test_notile.c` 先运行 `cute_matmul(..., M=128, N=64, K=128)` 到 `int32_t acc[128][64]`，再构造一个 `cute_post_call_t` 调 `cute_post_dequant_rope_bf16cvt`。
- `test_nopipeline.c` / `test_pipeline.c` 复用 `cute_tiled_matmul_no_pipeline_ex()` / `cute_tiled_matmul_pipeline_ex()`，`output_elem_bytes = sizeof(uint16_t)`。
- `a_scale` 继续使用 per-token scale，`b_scale` 使用 per-tensor weight scale；adapter 内不能再次叠加 `row0`。
- `rope_theta` 作为测试输入常量放在 test 或 generated header 里，manifest 可以记录但默认只 verify 输出。

golden：

- 新增 `golden/manual/fusion/matmul_dequant_rope_bf16cvt_m128_n64/`。
- generator 基于同一份 i8 matmul 输入，取前 64 列 acc，执行 `dequant + rope_bf16cvt`。
- 为减少数学实现漂移，优先复用 vector golden 的 RoPE/NVWA 参考逻辑；如果用 Python 生成，必须和 `cute_rope_bf16_tile` 的 theta/角度约定逐项对齐。

case.json：

- `golden`: `golden/manual/fusion/matmul_dequant_rope_bf16cvt_m128_n64/manifest.json`
- `verify.tensors`: 只验 `golden_output_bf16 -> output`
- `float_tolerance_percent`: 建议 `0.5`

验收：

- 三个 variant 输出全部在容差内。
- `notile` / `nopipeline` / `pipeline` 的输出一致性由共享 golden 间接保证。

### 7.2.2 `fusion_matmul_dequant_hadamard`

定位：

- 验证 FFN 类路径：`I8I8I32 matmul acc -> dequant -> hadamard(lhs) -> output + row_absmax`。
- Hadamard 是逐元素乘，天然支持 N 维切 tile。
- `row_absmax` 是按行归约结果；N=128 会被切成两个 tile col，adapter 必须把两个 tile 的 absmax 合并到同一个 `row_absmax[row]`。

建议 shape：

| 参数 | 值 |
|---|---|
| M | 128 |
| N | 128 |
| K | 128 |
| tile | 64x64, 2x2 tiles |
| matmul dtype | `CUTEDataTypeI8I8I32` |
| post-op input | int32 acc |
| lhs dtype | F32 |
| output dtype | F32 |
| extra output | F32 `row_absmax[M]` |

ctx：

```c
static cute_hadamard_ctx_t ctx = {
    .lhs = &lhs[0][0],
    .lhs_stride = N * sizeof(float),
    .output_absmax = row_absmax,
};
```

实现要点：

- `lhs[128][128]` 使用 deterministic 生成，避免把大数组手写进 test。
- `row_absmax[128]` 必须在运行前清零；因为 adapter 通过 `max(old, tile_absmax)` 合并两个 tile col。
- `cute_post_dequant_hadamard` 已经需要按 `row0/col0` 偏移 `lhs`，并按 `row0` 偏移 `output_absmax`；case 实现时重点回归这一点。
- `test_notile.c` 直接一次 post 128x128；`nopipeline/pipeline` 跑 2x2 tiles。

golden：

- 新增 `golden/manual/fusion/matmul_dequant_hadamard_m128_n128/`。
- manifest 至少包含：
  - `golden_output`
  - `golden_row_absmax`
  - 可选记录 `golden_lhs`，默认不作为 verify 输出。
- golden 计算顺序：
  1. `acc = matmul_i8_m128_n128_k128`
  2. `deq[row][col] = acc[row][col] * input_scale[row] * weight_scale[0]`
  3. `output[row][col] = deq[row][col] * lhs[row][col]`
  4. `row_absmax[row] = max(abs(output[row][:]))`

case.json：

```json
"verify": {
  "mode": "return_code_and_bit_exact",
  "tensors": [
    {
      "tensor": "golden_output",
      "symbol": "output",
      "float_tolerance_percent": 0.1
    },
    {
      "tensor": "golden_row_absmax",
      "symbol": "row_absmax",
      "float_tolerance_percent": 0.1
    }
  ]
}
```

验收：

- 三个 variant 同时验证 `output` 和 `row_absmax`。
- `pipeline` 模式下 `row_absmax` 不能受 tile 执行顺序影响。

### 7.2.3 `fusion_matmul_masked_softmax_kvscale_bf16cvt`

定位：

- 验证 attention scores 路径：`F32 scores -> kv_scale -> causal mask -> softmax -> BF16/F16 output`。
- 注意：这个 post-op 的输入类型是 `float *`，不是 int32 acc，所以不能直接接当前 i8 matmul acc。
- 注意：softmax 是按整行归一化。如果 `N=128` 被拆成两个 64-column tile，各 tile 独立 softmax 会得到错误语义。

第一版约束：

- 先做 `M = 128, N = 64, K = 64`，让每一行 softmax 的完整列范围落在一个 64-column tile 内。
- 这样 `nopipeline/pipeline` 仍有两个 tile row，可以覆盖 `row0`、mask row 偏移和 pipeline 行为；暂不覆盖跨 tile col 的 row-block softmax。

建议 shape：

| 参数 | 值 |
|---|---|
| M | 128 |
| N | 64 |
| K | 64 |
| tile | 64x64, 2x1 tiles |
| matmul dtype | 优先 `CUTEDataTypeF16F16F32` 或 `CUTEDataTypeBF16BF16F32` |
| post-op input | F32 scores |
| output dtype | F16/BF16 storage, manifest 先沿用 `F16` |
| kv_scale | `1 / sqrt(K)`，K=64 时为 `0.125f` |
| mask | causal bitmask, `max_ctx_len = 64` |

ctx：

```c
static const cute_softmax_ctx_t ctx = {
    .pos = 0,
    .bitmask = causal_mask,
    .max_ctx_len = 64,
    .kv_scale = 0.125f,
};
```

实现要点：

- 需要新增一套 F16/BF16 matmul 测试输入生成 helper，不能复用 i8 dequant helper。
- `test_notile.c` 跑一次 F16/BF16 matmul 到 `float scores[128][64]`，再一次 post softmax。
- `test_nopipeline.c` / `test_pipeline.c` 的 scratch buffer 类型应为 `float scratch[...][64][64]`，因为 post-op input 是 F32。
- 输出是 `uint16_t output[128][64]`，调用 `_ex()` 时 `output_elem_bytes = sizeof(uint16_t)`。
- `cute_post_masked_softmax_kvscale_bf16cvt` 需要把 `row0/col0` 传给 vector primitive；mask 的全局行语义由 `row0 + ctx.pos` 控制。

golden：

- 新增 `golden/manual/fusion/matmul_masked_softmax_kvscale_bf16cvt_m128_n64_k64/`。
- generator 计算：
  1. F16/BF16 QK matmul 得到 F32 scores。
  2. scores 乘 `kv_scale`。
  3. 应用 causal mask。
  4. row-wise softmax。
  5. 转 F16/BF16 存储。
- 优先复用 `fuse_masked_softmax_kvscale_bf16cvt` 的 vector golden 逻辑，避免 exp/softmax 近似漂移。

case.json：

- `verify.tensors`: `golden_output_bf16 -> output`
- `float_tolerance_percent`: 建议 `0.5`，如 softmax 近似差异较小可收紧。

后续 N=128 版本：

- 不能简单把当前 post-op 放到 2 个 `tile_j` 上跑。
- 需要新增 row-block softmax adapter：先收集同一 row-block 的两个 score tiles，做整行 max/sum，再写回两个 output tiles。
- 这个属于 `fuse_tensor_vec` 的第二阶段，不阻塞第一版 N=64 case。

---

## 8. 执行步骤

| 步骤 | 内容 |
|---|---|
| 1 | 创建 `cutelib/fusion/include/cute_fusion.h` |
| 2 | 实现 6 个 `cute_post_*` adapter |
| 3 | 添加 `cutelib_fusion` 和 `add_fusion_test_variant()` |
| 4 | 扩展 runner 支持 `build.variants` 和 `tests/fusion` |
| 5 | 添加 `fusion_matmul_dequant_silu` 三 variant case |
| 6 | 跑通三 binary 共用 golden 的闭环 |
| 7 | 扩展 resadd / bf16cvt |
| 8 | 扩展 RoPE：`fusion_matmul_dequant_rope_bf16cvt` |
| 9 | 扩展 Hadamard：`fusion_matmul_dequant_hadamard`，含 `row_absmax` 多 tensor verify |
| 10 | 扩展 Masked Softmax 第一版：`M128_N64_K64`，避免跨 tile softmax |
| 11 | 加入 `tests/smoke.yaml` 或新增/更新 `tests/fusion.yaml` |

验证命令：

```bash
cd /root/opencute/CUTE
cmake --build build -j$(nproc)
python3 tools/runner/cute-test.py --suite cute-sdk/tests/fusion.yaml --skip-build
```

---

## 9. Done Criteria

1. 每个 fusion case 一个 `case.json`，包含 `notile` / `nopipeline` / `pipeline` 三个 build variant。
2. 每个 variant 生成独立 `.riscv` binary。
3. 三个 binary 共用同一个 golden manifest。
4. runner 可以自动展开 variants，也可以单独跑某个 variant。
5. adapter 只调用 `cute_fuse_*_tile`，不重复数学逻辑。
6. `a_scale` 不重复叠加 `row0`。
7. 浮点输出允许按 `case.json` 配置的 `float_tolerance_percent` 通过，不要求 bit 级精确。

---

## 10. 风险与处理

| 风险 | 处理 |
|---|---|
| `case.json` variants 破坏现有 primitive/runtime case | runner 保持兼容：有 `build.source` 走旧逻辑，有 `build.variants` 才走新逻辑 |
| CMake 不读 JSON，variant 信息重复写两份 | 第一版允许手写 `add_fusion_test_variant()`，后续再生成 |
| variant 共享 golden 但输出 symbol 不一致 | 默认所有 test 使用同名 `output`，特殊情况在 variant 内覆盖 verify |
| adapter 和 vector primitive 语义漂移 | adapter 禁止复制数学逻辑，只调用 `cute_fuse_*_tile` |
| pipeline 输出与 notile 不一致时定位困难 | suite 支持单独跑 `case:variant` |
