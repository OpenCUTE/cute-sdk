# Phase C1 / Phase 3b: fusion case 组合实现计划

## Context

cute-sdk 需要把 Phase C0 中已经验证的 L2 单算子 primitive 组合成 L3 `cutelib/fusion` post-op，供 L1 tensor 的 `cute_tiled_matmul` pipeline 回调和 L4 layer 的 `cute_llama_block` 编排复用。

本计划建议在 [`phaseC0_single_primitive.md`](phaseC0_single_primitive.md) 和 [`phaseC1_vector_primitive.md`](phaseC1_vector_primitive.md) 之后执行：先把向量 primitive 接口收敛好，再由 fusion case 层把 primitive 组合成 `cute_tiled_matmul` 的 fused post-op，并接入不同调度模式。

在真正进入 fusion 之前，还需要先补一条 `notile` 参考软件路径：它面向同一份 golden，但不做 tile 级拆分，先验证“完成 cute matmul 之后再做向量后处理”的整块语义；之后再推进 `nopipeline` 和 `pipeline` 两种 tiled 版本。

Phase C1 的核心目标不是重写数学逻辑，而是做组合和适配：

1. 复用 Phase C0 已验证的 primitive，确保 bit-exact。
2. 去掉 `printf` / workload log / debug 分支。
3. 把不统一的 `void *ctx` 语义显式化，避免 model 层猜参数。
4. 让每个 fusion 函数只负责 glue：读取 `cute_post_call_t`、构造 primitive 参数、写回输出。

---

## 1. 迁移范围

### 1.1 Fusion post-op 范围

| 原函数 | 新 API | 用途 | 调用方式 |
|---|---|---|---|
| `fuse_ops_DEQUANT_ROPE_BF16CVRT` | `cute_fuse_dequant_rope_bf16cvt` | proj_q/proj_k: I32 -> F32 -> RoPE -> BF16 | tiled matmul post_op |
| `fuse_ops_DEQUANT_BF16CVRT_With_T` | `cute_fuse_dequant_bf16cvt` | proj_v: I32 -> F32 -> BF16，带 layout 转置写回 | tiled matmul post_op |
| `fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT` | `cute_fuse_masked_softmax_kvscale_bf16cvt` | scores: scale -> causal mask -> softmax -> BF16 | tiled matmul post_op 或单 tile post_op |
| `fuse_ops_DEQUANT_RESADD` | `cute_fuse_dequant_resadd` | proj_o/ffn_down: I32 -> F32 + residual | tiled matmul post_op |
| `fuse_ops_DEQUANT_SILU` | `cute_fuse_dequant_silu` | ffn_gate: I32 -> F32 -> SiLU | tiled matmul post_op |
| `fuse_ops_DEQUANT_HADAMARD` | `cute_fuse_dequant_hadamard` | ffn_up: I32 -> F32 -> hadamard + absmax | tiled matmul post_op |

`smoothquant`、`RMSnorm`、`RMSnorm_With_getabsmax_scale`、`vec_exp`、`vec_sin`、`vec_cos` 等不属于 fusion 组合层，放在 Phase C0 单算子 primitive 中实现。

### 1.2 三种软件实现模式

同一份 golden、同一套算子语义下，C1 相关的软件实现分成 3 个 `.c` 入口：

| 模式 | 行为 | 目的 |
|---|---|---|
| `notile` | matmul 完成后，直接对完整输出 buffer 做向量后处理，不拆 tile，也不做 overlap | 最粗的 reference 路径，先把 cute + vector 的端到端语义跑通 |
| `nopipeline` | 按 tile 拆分 matmul 与后处理，但 CPU post-op 与 CUTE 串行，不做 overlap | 验证 tile 边界、地址映射和每个 tile 的后处理结果 |
| `pipeline` | tile 拆分 + 双 buffer overlap | 最终高性能路径 |

这 3 个 `.c` 共用同一个 golden，只是调度粒度不同。

### 1.3 Phase C1 不做的事情

- 不重新实现 Phase C0 已完成的 primitive 数学逻辑。
- 不改变 Phase B 的 tile 调度策略。
- 不在 fusion 层管理模型级 buffer 生命周期。
- 不把 attention head 循环搬进 fusion 层。
- 不引入近似不同的新数学实现。
- 不支持非 llama3_1B shape 的泛化优化，除非原函数天然支持。

---

## 2. 文件结构

```
cutelib/fusion/include/
    cute_fusion.h       # fusion 回调类型、ctx 结构、6 个 matmul post_op

tests/fusion/
    test_fuse_dequant_rope_bf16cvt/
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    test_fuse_dequant_bf16cvt/
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    test_fuse_masked_softmax_kvscale_bf16cvt/
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    test_fuse_dequant_silu/
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    test_fuse_dequant_hadamard/
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
    test_fuse_dequant_resadd/
        test_notile.c
        test_nopipeline.c
        test_pipeline.c
```

如果项目继续保持 header-only 风格，`cutelib/fusion/include/*.h` 内全部使用 `static inline`。如果编译时间或重复符号成为问题，再把实现拆到 `cutelib/fusion/src/*.c`，但 Phase C1 第一版不做该拆分。

每个 `tests/fusion/<case>/` 建议共用一份 `case.json` / golden manifest，再挂三份软件入口：`test_notile.c`、`test_nopipeline.c`、`test_pipeline.c`。

---

## 3. API 定稿

### 3.1 与 L1 tensor 对齐的 post_op 签名

Phase B / L1 tensor 计划已经把回调收敛到 struct-based 形式：

```c
typedef void (*cute_post_op_fn)(const cute_post_call_t *call);
```

Phase C1 直接使用该签名，不再继续使用旧版：

```c
typedef void (*cute_fusion_fn)(
    void *cute_out, void *final_out,
    float *a_scale, float *b_scale,
    int dim_i, int dim_j,
    uint64_t cute_stride, uint64_t out_stride,
    void *ctx);
```

这样可以让 fusion 统一读取：

- `call->tile.src` / `call->tile.dst`
- `call->tile.rows` / `call->tile.cols`
- `call->tile.row0` / `call->tile.col0`
- `call->env.a_scale` / `call->env.b_scale`
- `call->user_ctx`

### 3.2 cute_fusion.h

```c
#ifndef CUTE_FUSION_H
#define CUTE_FUSION_H

#include <stdint.h>
#include <stddef.h>
#include "cute_ops.h"
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

void cute_fuse_dequant_rope_bf16cvt(const cute_post_call_t *call);
void cute_fuse_dequant_bf16cvt(const cute_post_call_t *call);
void cute_fuse_masked_softmax_kvscale_bf16cvt(const cute_post_call_t *call);
void cute_fuse_dequant_silu(const cute_post_call_t *call);
void cute_fuse_dequant_hadamard(const cute_post_call_t *call);
void cute_fuse_dequant_resadd(const cute_post_call_t *call);

#endif /* CUTE_FUSION_H */
```

---

## 4. 数据/布局约定

### 4.1 输入输出约定

| 字段 | 约定 |
|---|---|
| `call->tile.src` | CUTE matmul 输出 tile，I32 或 F32，按 `src_stride` 访问 |
| `call->tile.dst` | 最终输出 tile 起始地址，按 `dst_stride` 访问 |
| `call->env.a_scale` | per-token scale，指向当前 tile 起始 row 对应的 scale |
| `call->env.b_scale` | per-tensor weight scale，通常是 1 个 float |
| `call->tile.rows` | 当前 tile 有效 M，通常 64，尾 tile 可小于 64 |
| `call->tile.cols` | 当前 tile 有效 N，softmax 分支可为 `SEQ_LEN` |
| `call->tile.row0/col0` | 当前 tile 在全局输出矩阵中的起始坐标 |

### 4.2 ctx 约定

| fusion | `user_ctx` 类型 | 必填字段 |
|---|---|---|
| RoPE | `cute_rope_ctx_t *` | `pos`, `rope_theta`, `key_dim` |
| BF16 convert | 可为 `NULL` | 无 |
| Masked softmax | `cute_softmax_ctx_t *` | `pos`, `bitmask`, `max_ctx_len`, `kv_scale` |
| SiLU | 可为 `NULL` | 无 |
| Hadamard | `cute_hadamard_ctx_t *` | `lhs`, `lhs_stride`, `output_absmax` |
| ResAdd | `cute_resadd_ctx_t *` | `residual`, `residual_stride` |

### 4.3 与原实现保持一致的行为

- BF16 输出继续使用 `_Float16`/BF16 原路径中相同的转换方式。
- `FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT` 的 causal mask 和 `INV_SQRT_KEY_DIMENSION` scale 顺序保持不变。
- `fuse_ops_DEQUANT_BF16CVRT_With_T` 的 layout 写回保持与原实现一致，避免 proj_v/attention 后续地址计算变化。

---

## 5. 执行步骤

### Step 1: 准备 fusion 头文件骨架

创建：

- `cutelib/fusion/include/cute_fusion.h`

先只放 include guard、依赖 include、ctx struct 和函数声明。确保任意测试 include 这个头文件可以编译。

### Step 2: 组合简单 dequant fusion

优先迁移无复杂 ctx 的函数：

1. `cute_fuse_dequant_silu`
2. `cute_fuse_dequant_resadd`
3. `cute_fuse_dequant_bf16cvt`

这些函数用于验证 `cute_post_call_t` 到旧参数模型的映射是否正确。

每个函数内部第一步统一展开 `cute_post_call_t`，然后调用 Phase C0 primitive：

```c
int32_t *input = (int32_t *)call->tile.src;
void *output = call->tile.dst;
float *input_scale = call->env.a_scale;
float *weight_scale = call->env.b_scale;
int dim_i = call->tile.rows;
int dim_j = call->tile.cols;
uint64_t input_stride = call->tile.src_stride;
uint64_t output_stride = call->tile.dst_stride;
```

验收：

- 与原函数同输入 bit-exact。
- `row0/col0` 非 0 的 tile 写回地址正确。
- `a_scale` 使用当前 tile 行对应 scale，而不是全局 scale 起点。

### Step 3: 组合 RoPE fusion

迁移 `cute_fuse_dequant_rope_bf16cvt`，显式使用 `cute_rope_ctx_t`。

重点检查：

- 原 `pos` 是 `void *` 传入，Phase C1 改为 `ctx->pos`。
- `rope_theta` 不再读取 `llama3_1B.c` 全局数组，必须由 ctx 传入。
- `key_dim` 默认 64，但不要硬编码在函数内部。
- q/k 的 head 内 layout 写回与原实现一致。

测试建议：

- 用 `dim_i=64, dim_j=64` 的单 tile。
- 准备 `pos=0` 和 `pos>0` 两组 golden。
- q/k 共用同一测试函数，只换输出 buffer。

### Step 4: 组合 Hadamard fusion

迁移 `cute_fuse_dequant_hadamard`，显式使用 `cute_hadamard_ctx_t`。

重点检查：

- `lhs` 指向 gate SiLU 后的 F32 buffer。
- 当前函数输出复用 ffn_up F32 buffer，并写 `output_absmax[row]`。
- absmax 计算要按 token/row 聚合，不能按 tile 局部 col 分裂后覆盖。

如果 `FFN_DIMENSION=8192` 被拆成多个 N tile，`output_absmax[row]` 需要累积 max：

```c
output_absmax[row] = max(output_absmax[row], tile_absmax);
```

因此测试必须覆盖至少两个 N tile 的 FFN case，避免只测 64 列时漏掉该问题。

### Step 5: 组合 masked softmax fusion

迁移 `cute_fuse_masked_softmax_kvscale_bf16cvt`，显式使用 `cute_softmax_ctx_t`。

这是 Phase C1 风险最高的函数，按三步做：

1. 先搬原函数，保持 `dim_j != 64` 分支语义。
2. 用单 head、单 tile 的 `SEQ_LEN=64` golden 验证 softmax。
3. 再用 `SEQ_LEN=128` 验证 causal mask、row offset 和 BF16 写回。

重点检查：

- softmax 的 max/sum 归约范围必须是有效上下文长度。
- mask 地址来自 `ctx->bitmask`，不再读取 `bitmask_ptr` 全局。
- `kv_scale` 使用 ctx 字段，默认传 `INV_SQRT_KEY_DIMENSION`。
- 输出 stride 按 `scores_buf_q16[N_HEAD_Q][SEQ_LEN][SEQ_LEN]` 的实际布局传入。

Phase C1 第一版可以要求 softmax post_op 只支持 scores 的单 head 调用，不在 fusion 内处理多 head 循环。

### Step 6: CMake 集成

在顶层 `CMakeLists.txt` 添加：

```cmake
add_library(cutelib_fusion INTERFACE)
target_include_directories(cutelib_fusion INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/fusion/include
)
target_link_libraries(cutelib_fusion INTERFACE cutelib_primitive)
```

添加 fusion test helper：

```cmake
function(add_fusion_test case_dir)
    # follow existing runtime/tensor test pattern
endfunction()
```

### Step 7: 更新 smoke.yaml

把以下测试加入 `tests/smoke.yaml`：

- `fusion/test_fuse_dequant_silu`
- `fusion/test_fuse_dequant_resadd`
- `fusion/test_fuse_dequant_bf16cvt`
- `fusion/test_fuse_dequant_rope_bf16cvt`
- `fusion/test_fuse_dequant_hadamard`
- `fusion/test_fuse_masked_softmax_kvscale_bf16cvt`

每个 case 再由 helper 展开成 `notile` / `nopipeline` / `pipeline` 三个入口，不需要在 smoke 里手写 18 条名字。

---

## 6. 测试与 golden

### 6.1 golden 来源

优先级：

1. 直接复用 `llama3_1B.c` 固定输入和中间 buffer dump。
2. 如果已有 cutetest 中间结果头文件，直接 include。
3. 如果没有 golden，新增一个临时 dump 分支从原 `llama3_1B.c` 生成，不手写期望值。

同一个 golden 要同时喂给 `notile` / `nopipeline` / `pipeline` 三种软件入口，只是它们对 matmul 结果的消费方式不同。

### 6.2 测试矩阵

下面每个 case 都要分别跑 `notile`、`nopipeline`、`pipeline` 三种入口，golden 只维护一份。

| 测试 | 输入 shape | 验证内容 |
|---|---:|---|
| `test_fuse_dequant_silu` | 64 x 64 | dequant + SiLU bit-exact |
| `test_fuse_dequant_resadd` | 64 x 64 | dequant + residual add bit-exact |
| `test_fuse_dequant_bf16cvt` | 64 x 64 | dequant + BF16 convert + layout bit-exact |
| `test_fuse_dequant_rope_bf16cvt` | 64 x 64 | dequant + RoPE + BF16 bit-exact |
| `test_fuse_dequant_hadamard` | 64 x 128 | hadamard + absmax 跨 tile 累积 |
| `test_fuse_masked_softmax_kvscale_bf16cvt` | 64 x 128 | masked softmax + kv scale + BF16 bit-exact |

### 6.3 验证命令

```bash
cd /root/opencute/CUTE/cute-sdk
cmake --build build -j$(nproc)
python3 tools/runner/cute-test.py --suite cute-sdk/tests/smoke.yaml --skip-build
```

如果当前环境没有 RISC-V 仿真/硬件，最低验收为：

1. 所有 fusion test 编译通过。
2. 每个迁移函数与 `llama3_1B.c` 原函数逐行对照完成。
3. golden 生成脚本和数据路径在测试目录中固定下来。

---

## 7. 执行顺序

| 步骤 | 内容 | 依赖 | 预计耗时 |
|------|------|------|---------|
| 0 | 先完成 notile 参考软件 | 同一 golden 的整块后处理入口 | 30 min |
| 1 | 创建 fusion 头文件骨架 | L1 tensor API + Phase C0 primitive | 15 min |
| 2 | 组合 SiLU/ResAdd/BF16 简单 fusion | Phase C0 convert/elementwise | 45 min |
| 3 | 组合 RoPE fusion | Phase C0 convert/rope | 45 min |
| 4 | 组合 Hadamard fusion | Phase C0 convert/hadamard | 45 min |
| 5 | 组合 Masked Softmax fusion | Phase C0 softmax | 60 min |
| 6 | CMake + smoke.yaml 集成 | 全部测试目录 | 30 min |
| 7 | 跑 smoke / 对照 golden | build/runner 可用 | 60 min |

---

## 8. Done Criteria

Phase C1 完成时必须满足：

1. `cutelib_fusion` 可被 `cutelib_model` 链接。
2. 6 个 matmul post_op 全部使用 `cute_post_call_t`，无旧式散参数 wrapper 暴露给上层。
3. fusion 函数只做 primitive 组合和 post-op 适配，不重复实现数学核心。
4. 所有函数不依赖 `llama3_1B.c` 的全局变量。
5. 所有测试有固定 golden，并进入 smoke suite。
6. 与原 `llama3_1B.c` 对应 fused 函数 bit-exact。

---

## 9. 风险与处理

| 风险 | 处理 |
|------|------|
| L1 tensor 的 `cute_post_call_t` 还未实现或字段变化 | Phase C1 先以 Phase B 细化计划中的 struct 为准，字段变化时只调整 `cute_fusion.h` 映射层 |
| BF16 转换类型在编译器上表现不一致 | 保留原实现使用的类型和 intrinsic，不替换成手写 bit cast |
| Hadamard absmax 在多 N tile 下被覆盖 | `output_absmax[row] = max(old, tile_absmax)`，测试覆盖 128 列以上 |
| Softmax 分支 tiling 与普通 64x64 tiling 不同 | 第一版限定为 scores 单 head post_op，model 层按 head 调用 |
| RVV intrinsic 无法在 host 编译 | fusion tests 使用 RISC-V toolchain；host-only 单测不覆盖 RVV 实现 |
| golden 数据缺失 | 先新增 dump 工具/临时原实现测试，生成固定头文件后再迁移测试 |

---

## 10. 不在 Phase C1 范围内

- 不实现 `cute_llama_block`。
- 不做跨 layer buffer allocator。
- 不优化 softmax 算法复杂度。
- 不改变 `SEQ_LEN=128`、`EMBEDING_DIMENSION=2048`、`FFN_DIMENSION=8192` 等 llama3_1B 固定 shape。
- 不实现 smoothquant/RMSNorm/vec math primitive，它们属于 Phase C0。
