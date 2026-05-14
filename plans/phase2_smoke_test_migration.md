# Phase 2：三个 Smoke Test 移植到 cute-sdk

> 状态：**待执行**
> 前置依赖：Phase 1（memverify + golden 数据已就绪，三个 case 已全量 bit-exact 通过）
> 验收标准：一条命令从 cute-sdk 构建、仿真、验证三个 smoke test，统一输出 pass/fail。

---

## Context

Phase 1 已完成 memverify 闭环。三个 smoke test 已通过手动方式验证正确。现在需要把这三个 test 从 cutetest/ 正式迁移到 cute-sdk/，形成自动化的 build → run → verify 工作流，同时开始积攒 cutelib 库。

核心思路：**每个 test 迁移都是往 cutelib 里积攒 API 的机会**。test.c 变成薄调用层，组合封装沉淀为库。

## 依赖分层

```text
cuteisa/cute_isa_v1/instruction.h     # ISA 原子指令（已有，YGJK_INS_RRR 等）
        ↑
cutelib/runtime/cute_runtime.h        # 组合封装（本 Phase 积攒）
        ↑
tests/runtime/<case>/test.c           # 薄调用层：调库 + 等完成 + 读计数器
```

cute-sdk 完全自包含，不依赖 cutetest。

## 目标目录结构

```text
cute-sdk/
├── cuteisa/cute_isa_v1/               # ISA 基础层（已有）
├── cutelib/
│   └── runtime/
│       └── cute_runtime.h             # 组合封装：cute_matmul, cute_blockscale_matmul
├── tests/
│   └── runtime/
│       ├── runtime_matmul_i8_128_128_128_zeroinit/
│       │   ├── case.json
│       │   ├── test.c
│       │   └── matmul_value_mnk_128_128_128_zeroinit.h
│       ├── runtime_matmul_i8_128_128_128_zeroinit_transpose/
│       │   ├── case.json
│       │   ├── test.c
│       │   └── matmul_value_mnk_128_128_128_zeroinit_transpose.h
│       └── runtime_matmul_mxfp8e4m3_64_64_64_zeroinit/
│           ├── case.json
│           ├── test.c
│           └── matmul_value_mxfp8_mnk_64_64_64_zeroinit.h
├── golden/                            # （已有）
├── memverify/                         # （已有）
├── CMakeLists.txt                     # 根 CMake
├── cmake/
│   └── riscv-toolchain.cmake          # RISC-V 交叉编译 toolchain
└── run_test.py                        # test runner: build → simulate → verify
```

## 执行步骤

### 2.1 实现 cutelib/runtime/cute_runtime.h

从 cuteMarcoinstHelper.h 提取组合封装，基于 instruction.h 重写。

**API 对照**：

| 旧 helper | 新 API | 说明 |
|-----------|--------|------|
| `issue_cute_matmul_marco_inst(...)` | `cute_matmul(...)` | 配置 A/B/C/D + MNK + CONV + SEND |
| `issue_cute_blockscale_matmul_macro_inst(...)` | `cute_blockscale_matmul(...)` | 同上 + SCALE_A/B |
| `cute_marco_inst_fifo_finish_search()` | `CUTE_QUERY_MACRO_INST_FINISH()` | 已有，直接用 instruction.h |
| `cute_marco_inst_fifo_valid_search()` | `CUTE_QUERY_MACRO_INST_FIFO_FULL()` | 已有，直接用 |

**cute_runtime.h 骨架**：

```c
#ifndef CUTE_RUNTIME_H
#define CUTE_RUNTIME_H

#include "instruction.h"

// cute_matmul: D = A × B + C
// bias_mode: CUTE_BIAS_ZERO / CUTE_BIAS_ROW_REPEAT / CUTE_BIAS_FULL
static inline uint64_t cute_matmul(
    const void *a, uint64_t a_stride,
    const void *b, uint64_t b_stride,
    void *c, uint64_t c_stride,          // bias tensor
    void *d, uint64_t d_stride,          // output tensor
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t element_type,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t matmul_m_index)
{
    CUTE_CONFIG_TENSOR_A((uint64_t)a, a_stride);
    CUTE_CONFIG_TENSOR_B((uint64_t)b, b_stride);
    CUTE_CONFIG_TENSOR_C((uint64_t)c, c_stride);
    CUTE_CONFIG_TENSOR_D((uint64_t)d, d_stride);
    CUTE_CONFIG_TENSOR_DIM(m, n, k, 0);
    // CONFIG_CONV_PARAMS with element_type, bias_mode, transpose, ...
    CUTE_CONFIG_CONV_PARAMS(/* ... */);
    return CUTE_SEND_MACRO_INST();
}

// cute_blockscale_matmul: blockscale variant
static inline uint64_t cute_blockscale_matmul(
    const void *a, uint64_t a_stride,
    const void *b, uint64_t b_stride,
    const void *scale_a, const void *scale_b,
    void *c, uint64_t c_stride,
    void *d, uint64_t d_stride,
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t element_type,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t matmul_m_index)
{
    CUTE_CONFIG_TENSOR_A((uint64_t)a, a_stride);
    CUTE_CONFIG_TENSOR_B((uint64_t)b, b_stride);
    CUTE_CONFIG_SCALE_A((uint64_t)scale_a);
    CUTE_CONFIG_SCALE_B((uint64_t)scale_b);
    CUTE_CONFIG_TENSOR_C((uint64_t)c, c_stride);
    CUTE_CONFIG_TENSOR_D((uint64_t)d, d_stride);
    CUTE_CONFIG_TENSOR_DIM(m, n, k, 0);
    CUTE_CONFIG_CONV_PARAMS(/* ... */);
    return CUTE_SEND_MACRO_INST();
}

#endif // CUTE_RUNTIME_H
```

注意：cfgData2 的位移需要按 ISA 正确编码（Phase 1 已修复 cuteMarcoinstHelper.h 中的位移 bug，这里也要用正确的值）。

### 2.2 迁移 test.c（薄调用层）

每个 test.c 只做三件事：
1. 包含 cute_runtime.h + matmul_value_*.h
2. 调 `cute_matmul()` 或 `cute_blockscale_matmul()`
3. 等完成，读计数器

去掉所有 printf（或仅保留必要的状态输出）。去掉对 cutetest 的所有依赖。

**示例（i8 128x128）**：

```c
#include <stdint.h>
#include "cute_runtime.h"
#include "matmul_value_mnk_128_128_128_zeroinit.h"

int main(void) {
    uint64_t a_stride = APPLICATION_K * sizeof(a[0][0]);
    uint64_t b_stride = APPLICATION_K * sizeof(b[0][0]);
    uint64_t c_stride = APPLICATION_N * sizeof(c[0][0]);
    uint64_t d_stride = APPLICATION_N * sizeof(d[0][0]);

    cute_matmul(a, a_stride, b, b_stride, d, d_stride, c, c_stride,
                APPLICATION_M, APPLICATION_N, APPLICATION_K,
                CUTE_DATATYPE_I8I8I32, CUTE_BIAS_ZERO, 0, 0);

    while (!CUTE_QUERY_MACRO_INST_FINISH());

    return 0;
}
```

关键文件来源：

| Case | .h 数据来源 |
|------|-----------|
| i8_128_128_128_zeroinit | `cutetest/base_test/matmul_value_mnk_128_128_128_zeroinit.h` |
| i8_128_128_128_zeroinit_transpose | `cutetest/base_test/matmul_value_mnk_128_128_128_zeroinit_transpose.h` |
| mxfp8e4m3_64_64_64_zeroinit | `cutetest/datatype_mm_test/mxfp8e4m3/matmul_value_mxfp8_mnk_64_64_64_zeroinit.h` |

### 2.3 扩展 case.json

增加 `build` 和 `run` 字段：

```json
{
  "id": "runtime_matmul_i8_128_128_128_zeroinit",
  "op_ref": "ops/tensor/matmul.yaml",
  "level": "runtime",
  "build": {
    "source": "test.c",
    "target": "test.riscv"
  },
  "run": {
    "hwconfig": "cute4tops_shuttle512_d512_v512_m512_sysbus512_membus1_core_dramsim48",
    "trace_source": "run.out"
  },
  "golden": "golden/manual/tensor/matmul_i8_128_128_128_zeroinit/manifest.json",
  "verify": {
    "mode": "bit_exact",
    "tensor": "D"
  }
}
```

### 2.4 创建 CMake 构建系统

#### 层叠设计

每一层 SDK 库对应一个 CMake target，上层 `target_link_libraries` 依赖下层。
当前全为 header-only（`INTERFACE` 库），未来有 .c 实现时改为 `STATIC` 库，test 侧零改动。

```text
cuteisa (INTERFACE)          ← L0: ISA 原子指令
    ↑
cutelib_runtime (INTERFACE)  ← L1: runtime 组合封装
    ↑
cutelib_tensor (INTERFACE)   ← L2: tensor API（未来）
    ↑
cutelib_layer (INTERFACE)    ← L3: layer API（未来）
```

#### `cmake/riscv-toolchain.cmake`

```cmake
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(TOOLCHAIN_PREFIX "$ENV{CUTE_ROOT}/tool/riscv/bin/riscv64-unknown-elf-")
set(CMAKE_C_COMPILER   ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_OBJCOPY      ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
```

#### `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.15)
project(cute_sdk C)

# ---- 全局编译参数 ----
add_compile_options(
    -std=gnu99 -O3
    -march=rv64imafdcv -mabi=lp64d
    -specs=htif_nano.specs
    -fno-common -fno-builtin-printf -Wall
)
add_link_options(-static -specs=htif_nano.specs)

# ============================================================
# L0: cuteisa — ISA 原子指令（header-only）
# ============================================================
add_library(cuteisa INTERFACE)
target_include_directories(cuteisa INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuteisa/cute_isa_v1
)

# ============================================================
# L1: cutelib/runtime — 组合封装（header-only，依赖 cuteisa）
# ============================================================
add_library(cutelib_runtime INTERFACE)
target_include_directories(cutelib_runtime INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/runtime
)
target_link_libraries(cutelib_runtime INTERFACE cuteisa)

# # ============================================================
# # L2: cutelib/tensor — tensor API（未来）
# # ============================================================
# add_library(cutelib_tensor INTERFACE)
# target_include_directories(cutelib_tensor INTERFACE
#     ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/tensor
# )
# target_link_libraries(cutelib_tensor INTERFACE cutelib_runtime)

# ============================================================
# Tests: 每个 test case 一个 executable，链接对应层
# ============================================================

function(add_runtime_test case_dir)
    get_filename_component(case_name ${case_dir} NAME)
    set(test_src ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/${case_dir}/test.c)
    set(data_dir  ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime/${case_dir})
    add_executable(${case_name} ${test_src})
    target_include_directories(${case_name} PRIVATE ${data_dir})
    target_link_libraries(${case_name} PRIVATE cutelib_runtime)
    # 输出 test.riscv 而非默认的 case_name
    set_target_properties(${case_name} PROPERTIES OUTPUT_NAME test
        SUFFIX .riscv)
endfunction()

add_runtime_test(runtime_matmul_i8_128_128_128_zeroinit)
add_runtime_test(runtime_matmul_i8_128_128_128_zeroinit_transpose)
add_runtime_test(runtime_matmul_mxfp8e4m3_64_64_64_zeroinit)
```

#### 层叠扩展规则

新增层：3 行 CMake，`target_link_libraries` 自动传递下层 include。

```cmake
# 1. header-only 阶段（当前）
add_library(cutelib_tensor INTERFACE)
target_include_directories(cutelib_tensor INTERFACE cutelib/tensor)
target_link_libraries(cutelib_tensor INTERFACE cutelib_runtime)

# 2. 未来有 .c 实现时，改为 STATIC
# add_library(cutelib_tensor STATIC cutelib/tensor/cute_tensor.c)
# target_include_directories(cutelib_tensor PUBLIC cutelib/tensor)
# target_link_libraries(cutelib_tensor PUBLIC cutelib_runtime)
# test 侧零改动
```

新增 test：一行调用。

```cmake
add_runtime_test(runtime_matmul_new_case)
```

#### 构建命令

```bash
cd cute-sdk && mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/riscv-toolchain.cmake
make -j$(nproc)
```

### 2.5 实现 test runner

`cute-sdk/run_test.py`：

```
1. --case <id> 或 --all
2. build: cmake + make
3. run:   scripts/run-simulator-test.sh
4. verify: memverify.cute_memverify --manifest ... --trace ...
5. 报告 pass/fail
```

支持 `--skip-build`、`--skip-run`、`--trace <path>` 跳过任意阶段。

### 2.6 端到端验证

```bash
cd cute-sdk && python3 run_test.py --all
```

预期：

```
[runtime_matmul_i8_128_128_128_zeroinit]               BUILD OK  RUN OK  [PASS] 16384/16384
[runtime_matmul_i8_128_128_128_zeroinit_transpose]      BUILD OK  RUN OK  [PASS] 16384/16384
[runtime_matmul_mxfp8e4m3_64_64_64_zeroinit]            BUILD OK  RUN OK  [PASS] 4096/4096
```

## 关键文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 创建 | `cutelib/runtime/cute_runtime.h` | 基于 instruction.h 的组合封装 |
| 创建 | `tests/runtime/<case>/test.c` | 薄调用层，调 cute_runtime.h |
| 复制 | `tests/runtime/<case>/matmul_value_*.h` | 测试数据 |
| 修改 | `tests/runtime/<case>/case.json` | 增加 build/run 字段 |
| 创建 | `CMakeLists.txt` | 根 CMake（层叠 target + test 函数） |
| 创建 | `cmake/riscv-toolchain.cmake` | RISC-V 交叉编译 toolchain |
| 创建 | `run_test.py` | test runner |

## 不在 Phase 2 范围内

- 不做 CI 集成
- 不新增 smoke case
- cute_runtime.h 保持 header-only，不做 .c/.o 分离（函数都是 static inline，无链接需求）
