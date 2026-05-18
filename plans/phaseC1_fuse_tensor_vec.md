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
```

后续再扩展 RoPE、Hadamard、Masked Softmax。

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

## 8. 执行步骤

| 步骤 | 内容 |
|---|---|
| 1 | 创建 `cutelib/fusion/include/cute_fusion.h` |
| 2 | 实现 6 个 `cute_post_*` adapter |
| 3 | 添加 `cutelib_fusion` 和 `add_fusion_test_variant()` |
| 4 | 扩展 runner 支持 `build.variants` 和 `tests/fusion` |
| 5 | 添加 `fusion_matmul_dequant_silu` 三 variant case |
| 6 | 跑通三 binary 共用 golden 的闭环 |
| 7 | 扩展 resadd / bf16cvt / rope / hadamard / masked softmax |
| 8 | 加入 `tests/smoke.yaml` 或新增 `tests/fusion.yaml` |

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
7. 所有 fusion variants bit-exact 通过。

---

## 10. 风险与处理

| 风险 | 处理 |
|---|---|
| `case.json` variants 破坏现有 primitive/runtime case | runner 保持兼容：有 `build.source` 走旧逻辑，有 `build.variants` 才走新逻辑 |
| CMake 不读 JSON，variant 信息重复写两份 | 第一版允许手写 `add_fusion_test_variant()`，后续再生成 |
| variant 共享 golden 但输出 symbol 不一致 | 默认所有 test 使用同名 `output`，特殊情况在 variant 内覆盖 verify |
| adapter 和 vector primitive 语义漂移 | adapter 禁止复制数学逻辑，只调用 `cute_fuse_*_tile` |
| pipeline 输出与 notile 不一致时定位困难 | suite 支持单独跑 `case:variant` |
