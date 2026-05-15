# Phase B: L2 cutelib/tensor 实现计划

## Context

cute-sdk 需要在 L1 runtime 之上加一层 tensor 抽象，提供 `cute_tensor_t` 描述符 + 单 tile matmul + tiled matmul with pipeline。这是 llama 模型迁移计划 Phase B，为后续 fusion/model 层打基础。

迁移计划中有两个 bug 需要修复：
1. cute_matmul_op 的 c/d 参数混淆（c 实际是 bias，d 是 output）
2. tiled_matmul 循环缺少第一个 tile 的 issue（会 deadlock）

## 文件结构

```
cutelib/tensor/include/
    cute_tensor.h      # Tensor 描述符 + 常量 + 辅助函数
    cute_ops.h         # cute_matmul_op + cute_tiled_matmul

tests/tensor/
    tensor_matmul_i8_128_128_128_zeroinit/          # 单次调用等价性
    tensor_matmul_i8_128_128_128_zeroinit_transpose/ # transpose 路径
    tensor_matmul_mxfp8e4m3_64_64_64_zeroinit/      # blockscale
    tensor_matmul_i8_tiled_128x128_fifo/             # tiled pipeline
```

## Step 1: cute_tensor.h

创建 `cutelib/tensor/include/cute_tensor.h`：

- `CUTE_BIAS_ZERO=1`, `CUTE_BIAS_ROW_REPEAT=2`, `CUTE_BIAS_FULL=3`
- `CUTE_SCALE_NONE=0`, `CUTE_SCALE_PERTOKEN_A_PERTENSOR_B=1`
- `CUTE_TILE_M=64`, `CUTE_TILE_N=64`
- `cute_tensor_t` 结构体：`{data, stride, rows, cols, dtype}`
- `cute_stride(cols, dtype)` — 按 dtype 算输入行步长
- `cute_output_stride(cols)` — 输出固定 `cols * 4`

## Step 2: cute_ops.h

创建 `cutelib/tensor/include/cute_ops.h`：

### cute_matmul_op — 单次 matmul wrapper

```c
static inline uint64_t cute_matmul_op(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    const cute_tensor_t *bias,    // 注意命名：bias，不用 c
    const cute_tensor_t *output,  // 注意命名：output，不用 d
    uint64_t bias_mode, uint64_t transpose, uint64_t m_index)
```

直接映射到 `cute_matmul()`，不限制 M/N <= 64（硬件内部自动 tile）。

### cute_blockscale_matmul_op — blockscale wrapper

类似，映射到 `cute_blockscale_matmul()`。

### cute_fusion_fn — fusion 回调类型

```c
typedef void (*cute_fusion_fn)(
    void *cute_buf, void *final_out,
    float *a_scale, float *b_scale,
    int dim_i, int dim_j,
    uint64_t cute_stride, uint64_t out_stride,
    void *ctx);
```

### cute_tiled_matmul — 核心 tiled matmul

```c
static inline void cute_tiled_matmul(
    const cute_tensor_t *a,      // [M, K]
    const cute_tensor_t *b,      // [K, N]
    void *output,                // 最终输出 [M, N]
    uint64_t output_stride,
    const cute_tensor_t *bias,   // bias tensor（zero-init 时传有效地址即可）
    float *a_scale, float *b_scale,
    int scale_type, int bias_mode, int transpose,
    void *double_buf,            // >= CUTE_TILE_M * CUTE_TILE_N * 4 字节（post_op 时用）
    cute_fusion_fn post_op,      // NULL = 直接写入 output
    void *post_ctx)
```

**两种路径**：

1. **post_op == NULL（直接写入）**：CUTE 直接写到 output 对应象限，不需要 double_buf。
2. **post_op != NULL（单 buffer pipeline）**：CUTE 写到 double_buf，wait 完成后 CPU 调 post_op 处理 double_buf → output，然后再 issue 下一个 tile（写同一个 double_buf）。

**Pipeline 算法**（修复迁移计划的 bug）：

```
线性化 tiles: (0,0), (0,1), (1,0), (1,1) ...
total = tile_i * tile_j

1. Issue tile 0
2. for n = 1 .. total-1:
     wait(prev_tid)
     issue tile n
     if post_op: post_op(buf -> output_quadrant(prev))
3. wait(last_tid)
   if post_op: post_op(buf -> output_quadrant(last))
```

overlap 发生在：CUTE 算 tile N+1 的同时，CPU 做 tile N 的 post_op。

**关键修复**：
- 先 issue 第一个 tile，然后 wait+issue+post_op 循环（修复计划中缺少首 tile issue 的 bug）
- bias/output 命名替代 c/d（消除歧义）
- 只需 1 个 double_buf（深度 1 pipeline，够用且简单）

## Step 3: CMake 集成

在 `CMakeLists.txt` 中添加：
```cmake
add_library(cutelib_tensor INTERFACE)
target_include_directories(cutelib_tensor INTERFACE cutelib/tensor/include)
target_link_libraries(cutelib_tensor INTERFACE cutelib_runtime)

function(add_tensor_test case_dir) ... endfunction()
```

## Step 4: 测试

| 测试 | 验证点 | golden 来源 |
|------|--------|------------|
| tensor_matmul_i8_128_128_128_zeroinit | cute_matmul_op 等价于 runtime | 复用 runtime 的 golden |
| tensor_matmul_i8_128_128_128_zeroinit_transpose | transpose 路径 | 复用 runtime transpose 的 golden |
| tensor_matmul_mxfp8e4m3_64_64_64_zeroinit | blockscale 路径 | 复用 runtime 的 golden |
| tensor_matmul_i8_tiled_128x128_fifo | cute_tiled_matmul 结果 = 单次 matmul | 复用 128x128 golden |

测试数据通过相对路径引用 runtime 测试的 .h 文件。

## Step 5: 更新 smoke.yaml

添加 tensor 级测试到 smoke suite。

## 关键文件

- `cutelib/runtime/cute_runtime.h` — L1 API（cute_matmul, cute_wait_task）
- `tests/runtime/runtime_matmul_i8_tiled_128x128_fifo/test.c` — 已验证的 tiled pipeline 模式
- `cuteisa/cute_isa_v1/cute_fpe.h` — dtype 常量
- `CMakeLists.txt` — 构建系统

## 验证方式

1. tensor 级单次调用测试的 memverify 输出必须与对应 runtime 测试 bit-exact
2. tiled 测试的 memverify 输出必须与 runtime_matmul_i8_128_128_128_zeroinit bit-exact
3. `python3 tools/runner/cute-test.py --suite cute-sdk/tests/smoke.yaml --skip-build` 全部 PASS
