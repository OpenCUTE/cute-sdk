# cutetest/llama → cute-sdk 全量迁移计划

> 目标：用 cute-sdk 的分层 lib 重写 `llama3_1B.c`，验证端到端 correctness。
> 参考实现：`cutetest/transformer_test/llama/llama3_1B.c`

---

## 1. 目标架构

```text
L0 cuteisa                          ← ISA 原子指令（已完成）
    ↑
L1 cutelib/runtime                  ← CUTE 硬件交互原语（Phase 2 已部分完成，需扩展）
    ↑
L2 cutelib/tensor                   ← Tensor 描述 + Tiled Matmul + Pipeline 调度
    ↑
L3 cutelib/fusion                   ← CPU 端后处理（dequant / RoPE / softmax / SiLU / quant / rmsnorm）
    ↑
L4 cutelib/model                    ← Transformer Block 编排（llama_block）
```

验收标准：`cute_llama_block()` 的输出与 `llama3_1B.c` 的 `llama_block()` bit-exact 一致。

---

## 2. 当前状态

### 已完成

| 层 | 状态 | 说明 |
|----|------|------|
| L0 cuteisa | **完成** | `instruction.h` + `cute_fpe.h` 全部 ISA wrapper |
| L1 cutelib/runtime | **部分完成** | `cute_matmul()` + `cute_blockscale_matmul()` 可用 |
| L2-L4 | **未开始** | — |

### L1 缺口分析

`llama3_1B.c` 的 `CUTE_TASK_END(task_id)` 使用了以下原语，当前 `cute_runtime.h` 未封装：

| llama3_1B.c 调用 | instruction.h 对应 | cute_runtime.h 状态 |
|---|---|---|
| `cute_marco_inst_fifo_finish_search()` | `CUTE_QUERY_MACRO_INST_FINISH()` | **未封装** |
| `cute_marco_inst_fifo_dequeue()` | `CUTE_CLEAR_INST()` | **未封装** |
| `cute_marco_inst_fifo_full_search()` | `CUTE_QUERY_MACRO_INST_FIFO_FULL()` | **未封装** |
| `cute_marco_inst_fifo_valid_search()` | `CUTE_QUERY_MACRO_INST_FIFO_INFO()` | **未封装** |
| `cute_marco_inst_fifo_get_finish_tail_fifoindex()` | `CUTE_QUERY_INST()` | **未封装** |

没有这些原语，tensor 层无法实现 pipeline 调度（逐 tile issue → wait → dequeue → issue next）。

---

## 3. llama3_1B.c 执行模式分析

### 3.1 matmul_cute 的三层职责

`matmul_cute()` 是 llama3_1B.c 的核心函数（line 669-849），它交织了三个职责：

```text
职责 1 [tensor]: 大 shape → 64×64 tile 拆分
职责 2 [runtime]: 逐 tile issue CUTE 指令，pipeline 调度
职责 3 [fusion]:  等 tile 完成后，CPU 端做后处理（dequant/softmax/silu 等）
```

执行时序：

```text
CUTE 加速器:  [tile0 matmul] [tile1 matmul] [tile2 matmul] [tile3 matmul] ...
CPU (fusion):                 [tile0 fusion] [tile1 fusion] [tile2 fusion] ...
                               ↑── overlap ──↑
```

### 3.2 pipeline 模式伪代码

```c
// matmul_cute 的核心循环（简化）
issue(tile_0);                           // 发 tile 0
for (each remaining tile) {
    wait(prev_task_id); dequeue();       // 等 tile N 完成
    issue(next_tile);                    // 发 tile N+1（与 CPU 并行）
    fusion(prev_result, output);         // CPU 做 tile N 后处理
}
wait(last_task_id); dequeue();
fusion(last_result, output);
```

### 3.3 fusion 函数清单

`matmul_cute()` 的 `after_ops` 参数触发 6 种后处理：

| after_ops 常量 | 函数 | 用途 |
|---|---|---|
| `FUSE_DEQUANT_ROPE_BF16CVRT` (1025) | `fuse_ops_DEQUANT_ROPE_BF16CVRT` | proj_q, proj_k: I32→F32→RoPE→BF16 |
| `FUSE_DEQUANT_BF16CVRT` (1026) | `fuse_ops_DEQUANT_BF16CVRT_With_T` | proj_v: I32→F32→BF16 |
| `FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT` (1027) | `fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT` | scores: scale→masked softmax→BF16 |
| `FUSE_DEQUANT_SILU` (1029) | `fuse_ops_DEQUANT_SILU` | ffn_gate: I32→F32→SiLU |
| `FUSE_DEQUANT_HADAMARD` (1030) | `fuse_ops_DEQUANT_HADAMARD` | ffn_up: I32→F32→elementwise 乘 |
| `FUSE_DEQUANT_RESADD` (1031) | `fuse_ops_DEQUANT_RESADD` | proj_o/ffn_down: I32→F32+残差 |

加上两个独立操作（不在 matmul_cute 内，但在 llama_block 中调用）：

| 函数 | 用途 |
|---|---|
| `smoothquant()` | F32→INT8 per-token 量化（3 处调用） |
| `RMSnorm_With_getabsmax_scale()` | RMS 归一化 + 提取 absmax（2 处调用） |

### 3.4 llama_block 调用序列

```text
llama_block:

  [1] RMSnorm + smoothquant           (input → hidden_states_q8)

  [2] tiled_matmul(proj_q) + fuse(dequant+rope+bf16)    SEQ_LEN × EMBED × EMBED, I8×I8→I32
  [3] tiled_matmul(proj_k) + fuse(dequant+rope+bf16)    SEQ_LEN × EMBED/4 × EMBED, I8×I8→I32
  [4] tiled_matmul(proj_v) + fuse(dequant+bf16)         SEQ_LEN × EMBED/4 × EMBED, I8×I8→I32

  [5] for head in 0..31:
        tiled_matmul(scores) + fuse(masked_softmax+bf16)  SEQ_LEN × SEQ_LEN × KEY_DIM, BF16×BF16→F32

  [6] for head in 0..31:
        tiled_matmul(attn)   + NO_ACTIVATION              SEQ_LEN × VALUE_DIM × SEQ_LEN, BF16×BF16→F32

  [7] smoothquant                         (attn → attn_q8)

  [8] tiled_matmul(proj_o) + fuse(dequant+resadd)        SEQ_LEN × EMBED × EMBED, I8×I8→I32

  [9] RMSnorm + smoothquant              (proj_o → proj_o_q8)

  [10] tiled_matmul(ffn_gate) + fuse(dequant+silu)       SEQ_LEN × FFN × EMBED, I8×I8→I32
  [11] tiled_matmul(ffn_up) + fuse(dequant+hadamard)     SEQ_LEN × FFN × EMBED, I8×I8→I32

  [12] smoothquant                        (hadamard → ffn_up_q8)

  [13] tiled_matmul(ffn_down) + fuse(dequant+resadd)     SEQ_LEN × EMBED × FFN, I8×I8→I32
```

共 13 步，8 次 tiled_matmul（其中 6 次带 fusion 回调），3 次 smoothquant，2 次 rmsnorm。

---

## 4. 分层执行计划

### Phase A：L1 runtime 扩展（前置，无测试依赖）

> 目标：补齐 pipeline 调度所需的全部 runtime 原语。

#### A.1 在 cute_runtime.h 中新增 FIFO 管理 wrapper

```c
// 非阻塞：返回已完成任务的 bitmask
static inline uint64_t cute_query_finish(void) {
    return CUTE_QUERY_MACRO_INST_FINISH();
}

// 非阻塞：FIFO 是否已满（1=满）
static inline int cute_fifo_full(void) {
    return CUTE_QUERY_MACRO_INST_FIFO_FULL() != 0;
}

// 非阻塞：FIFO 当前占用
static inline uint64_t cute_fifo_info(void) {
    return CUTE_QUERY_MACRO_INST_FIFO_INFO();
}

// 阻塞：等待特定 task_id 完成，然后出队
// 等价于 llama3_1B.c 的 CUTE_TASK_END(task_id)
//
// 【FIFO 顺序约束】
// CUTE 宏指令 FIFO 严格先入先出。CUTE_CLEAR_INST() 无参数，永远出队队首。
// 因此软件必须保证：
//   1. wait+dequeue 严格按 issue 顺序调用（不能跳过中间 task）
//   2. task_id 必须是当前 FIFO 中最早进入的那条未完成指令
//   3. 先 issue 的指令必定先完成，不存在乱序完成
static inline void cute_wait_task(uint64_t task_id) {
    uint64_t mask = 1UL << task_id;
    while (!(CUTE_QUERY_MACRO_INST_FINISH() & mask))
        ;
    CUTE_CLEAR_INST();
}

// 出队队首已完成任务（不等待）
// 同样受 FIFO 顺序约束：只能出队当前队首（即最早 issue 的那条）
static inline void cute_dequeue(void) {
    CUTE_CLEAR_INST();
}

// 查询已完成任务的尾编号位置
static inline uint64_t cute_query_inst_tail(void) {
    return CUTE_QUERY_INST();
}
```

**验证**：编译通过。语义与 `cuteMarcoinstHelper.h` 中的对应函数等价。

#### A.2 新增 cute_send 独立封装

```c
// 单独发送已配置的宏指令，返回 task_id
// 用于需要拆分 config/send 的场景
static inline uint64_t cute_send(void) {
    return CUTE_SEND_MACRO_INST();
}
```

当前 `cute_matmul()` 已经内部调用了 `CUTE_SEND_MACRO_INST()` 并返回 task_id，但有些场景需要先 config 再 send。

### Phase B：L2 cutelib/tensor

> 目标：提供 `cute_tensor_t` 描述符 + 单 tile matmul + tiled matmul with pipeline。

#### B.1 定义 cute_tensor.h

文件：`cutelib/tensor/include/cute_tensor.h`

```c
#ifndef CUTE_TENSOR_H
#define CUTE_TENSOR_H

#include <stdint.h>
#include "cute_fpe.h"

// ---- Bias 加载模式 ----
#define CUTE_BIAS_ZERO        1
#define CUTE_BIAS_ROW_REPEAT  2
#define CUTE_BIAS_FULL        3

// ---- Scale 类型 ----
#define CUTE_SCALE_NONE                   0
#define CUTE_SCALE_PERTOKEN_A_PERTENSOR_B 1

// ---- Tensor 描述符 ----
typedef struct {
    void    *data;
    uint64_t stride;    // 行步长 (byte)
    uint64_t rows;      // M
    uint64_t cols;      // N 或 K
    uint64_t dtype;     // ElementDataType
} cute_tensor_t;

// ---- 辅助：计算行步长 ----
static inline uint64_t cute_stride(uint64_t cols, uint64_t dtype) {
    uint64_t elem_bytes;
    switch (dtype) {
        case CUTEDataTypeI8I8I32:
        case CUTEDataTypeI8U8I32:
        case CUTEDataTypeU8I8I32:
        case CUTEDataTypeU8U8I32:
            elem_bytes = 1; break;
        case CUTEDataTypeF16F16F32:
        case CUTEDataTypeBF16BF16F32:
            elem_bytes = 2; break;
        case CUTEDataTypeTF32TF32F32:
            elem_bytes = 4; break;
        default:
            elem_bytes = 1; break;   // blockscale dtypes, element 级也是 1 byte
    }
    return cols * elem_bytes;
}

// 输出 tensor stride（始终为 I32 或 F32，4 byte）
static inline uint64_t cute_output_stride(uint64_t cols) {
    return cols * 4;
}

// ---- 硬件 tile 尺寸 ----
#define CUTE_TILE_M 64
#define CUTE_TILE_N 64

#endif /* CUTE_TENSOR_H */
```

#### B.2 定义 cute_ops.h

文件：`cutelib/tensor/include/cute_ops.h`

**单 tile matmul**：直接调 runtime，要求 shape <= 64×64。

```c
// D = A × B + C，单 tile
// A: [M, K], B: [K, N], C: [M, N], D: [M, N]
static inline uint64_t cute_matmul_op(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    const cute_tensor_t *c,
    const cute_tensor_t *d,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t m_index)
{
    return cute_matmul(
        a->data, a->stride,
        b->data, b->stride,
        c->data, c->stride,
        d->data, d->stride,
        a->rows, d->cols, a->cols,
        a->dtype, bias_mode, transpose, m_index);
}

// Blockscale 变体
static inline uint64_t cute_blockscale_matmul_op(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    const cute_tensor_t *c,
    const cute_tensor_t *d,
    const void *scale_a, const void *scale_b,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t m_index)
{
    return cute_blockscale_matmul(
        a->data, a->stride,
        b->data, b->stride,
        scale_a, scale_b,
        c->data, c->stride,
        d->data, d->stride,
        a->rows, d->cols, a->cols,
        a->dtype, bias_mode, transpose, m_index);
}
```

**Tiled matmul with pipeline**：核心函数。

```c
// ---- Fusion 回调类型 ----
// 对每个完成的 tile 调用
//   cute_out:     CUTE 输出 buffer（64×64 I32/F32）
//   final_out:    写回最终 tensor 的目标地址
//   a_scale:      A 的 per-token scale（可为 NULL）
//   b_scale:      B 的 per-tensor scale（可为 NULL）
//   dim_i, dim_j: tile 尺寸（通常 64×64）
//   cute_stride:  cute_out 的行步长
//   out_stride:   final_out 的行步长
//   ctx:          回调上下文（如 pos、output_scale 指针等）
typedef void (*cute_fusion_fn)(
    void *cute_out, void *final_out,
    float *a_scale, float *b_scale,
    int dim_i, int dim_j,
    uint64_t cute_stride, uint64_t out_stride,
    void *ctx);

// ---- Tiled Matmul ----
// 将大 shape 拆成 64×64 tile，pipeline 调度
// double_buf: 至少 2 个缓冲区（推荐 4 个），每个 >= CUTE_TILE_M * CUTE_TILE_N * 4
// n_buf:      double_buf 数组长度
// post_op:    后处理回调（可为 NULL = NO_ACTIVATION）
// post_ctx:   传给 post_op 的 ctx
static inline void cute_tiled_matmul(
    const cute_tensor_t *a,     // [M, K]
    const cute_tensor_t *b,     // [K, N]
    void *output,               // 最终输出地址 [M, N]
    uint64_t output_stride,     // output 行步长
    float *a_scale,             // A scale（per-token），可为 NULL
    float *b_scale,             // B scale（per-tensor），可为 NULL
    int scale_type,             // CUTE_SCALE_NONE / CUTE_SCALE_PERTOKEN_A_PERTENSOR_B
    int bias_mode,              // CUTE_BIAS_ZERO 等
    int transpose,
    void **double_buf,          // pipeline 缓冲区数组
    int n_buf,                  // 缓冲区数量（>= 2）
    cute_fusion_fn post_op,     // 后处理回调
    void *post_ctx)             // 回调上下文
{
    int M = a->rows, N = b->cols, K = a->cols;
    int tile_i = M / CUTE_TILE_M;
    int tile_j = N / CUTE_TILE_N;

    // 特殊路径：shape <= 64×64，不需要 tiling
    if (tile_i <= 1 && tile_j <= 1) {
        cute_tensor_t tile_a = {a->data, a->stride, M, K, a->dtype};
        cute_tensor_t tile_b = {b->data, b->stride, K, N, a->dtype};
        cute_tensor_t tile_c = {double_buf[0], CUTE_TILE_N * 4, M, N, a->dtype};
        cute_tensor_t tile_d = {NULL, 0, 0, 0, 0};

        uint64_t tid = cute_matmul_op(&tile_a, &tile_b, &tile_c, &tile_d,
                                       bias_mode, transpose, 0);
        cute_wait_task(tid);

        if (post_op) {
            post_op(double_buf[0], output, a_scale, b_scale,
                    M, N, CUTE_TILE_N * 4, output_stride, post_ctx);
        } else {
            // 直接拷贝到 output
            // ...
        }
        return;
    }

    // Tiled pipeline 路径
    int buf_idx = 0;
    uint64_t prev_tid = 0;
    int prev_i = 0, prev_j = 0;
    int first = 1;

    for (int i = 0; i < tile_i; i++) {
        for (int j = (i == 0 ? 1 : 0); j < tile_j; j++) {
            // Wait previous + dequeue
            cute_wait_task(prev_tid);

            // Issue current tile
            void *tile_a_data = (char*)a->data + i * CUTE_TILE_M * a->stride;
            void *tile_b_data = (char*)b->data + j * CUTE_TILE_N * b->stride;
            int next_buf = (buf_idx + 1) % n_buf;

            cute_tensor_t ta = {tile_a_data, a->stride, CUTE_TILE_M, K, a->dtype};
            cute_tensor_t tb = {tile_b_data, b->stride, K, CUTE_TILE_N, a->dtype};
            cute_tensor_t tc = {double_buf[next_buf], CUTE_TILE_N * 4, CUTE_TILE_M, CUTE_TILE_N, a->dtype};
            cute_tensor_t td = {NULL, 0, 0, 0, 0};

            prev_tid = cute_matmul_op(&ta, &tb, &tc, &td, bias_mode, transpose, 0);

            // Run post_op on previous tile (overlaps with CUTE compute)
            if (post_op && !first) {
                void *dst = (char*)output +
                    (transpose ? prev_j : prev_i) * CUTE_TILE_M * output_stride +
                    (transpose ? prev_i : prev_j) * CUTE_TILE_N * cute_output_stride(1);
                post_op(double_buf[buf_idx], dst,
                        a_scale + prev_i * CUTE_TILE_M, b_scale,
                        CUTE_TILE_M, CUTE_TILE_N,
                        CUTE_TILE_N * 4, output_stride, post_ctx);
            }

            buf_idx = next_buf;
            prev_i = i; prev_j = j;
            first = 0;
        }
    }

    // Wait last + post_op
    cute_wait_task(prev_tid);
    if (post_op) {
        void *dst = (char*)output +
            (transpose ? prev_j : prev_i) * CUTE_TILE_M * output_stride +
            (transpose ? prev_i : prev_j) * CUTE_TILE_N * cute_output_stride(1);
        post_op(double_buf[buf_idx], dst,
                a_scale + prev_i * CUTE_TILE_M, b_scale,
                CUTE_TILE_M, CUTE_TILE_N,
                CUTE_TILE_N * 4, output_stride, post_ctx);
    }
}
```

**注意**：上面是参考实现骨架。特殊路径 `FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVRT` 的 tiling 模式不同（只按 M 维 tile，N 维不拆），需要在实际实现时处理。详见 llama3_1B.c line 786-841 的 `dim_j != 64` 分支。

#### B.3 CMake 集成

```cmake
# L2: cutelib/tensor — tensor API（header-only）
add_library(cutelib_tensor INTERFACE)
target_include_directories(cutelib_tensor INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/tensor/include
)
target_link_libraries(cutelib_tensor INTERFACE cutelib_runtime)
```

#### B.4 迁移 tensor 级测试

从 Phase 2 的 runtime 测试迁移，验证 tensor API 等价性：

| Case | 验证点 |
|---|---|
| `tensor_matmul_i8_128_128_128_zeroinit` | INT8 单 tile，与 runtime 等价 |
| `tensor_matmul_i8_128_128_128_zeroinit_transpose` | transpose 路径 |
| `tensor_matmul_mxfp8e4m3_64_64_64_zeroinit` | blockscale + 单 tile |
| `tensor_matmul_i8_256_256_256_zeroinit` | **新增**，验证 tiled 路径（2×2=4 tiles） |

tiled 测试需要 golden 数据（256×256 的 I8 matmul），可从 cutetest 导入或用 `get_mattest_value.py` 生成。

### Phase C：L3 cutelib/fusion

> 目标：将 llama3_1B.c 中 7 个 fusion 函数 + 2 个独立操作迁入 cutelib/fusion/。

#### C.1 目录结构

```text
cutelib/fusion/
└── include/
    ├── cute_fusion.h       # Fusion 回调统一声明
    ├── cute_quant.h        # smoothquant + rmsnorm
    └── cute_vec_math.h     # 底层向量数学：vec_exp, vec_sin, vec_cos（internal）
```

#### C.2 fusion 回调函数

7 个函数全部从 llama3_1B.c 搬入，去除 printf 和 debug 代码，统一为 `cute_fusion_fn` 签名：

```c
// cute_fusion.h

// 6 个 fusion 回调（与 cute_tiled_matmul 配合使用）
cute_fusion_fn cute_fuse_dequant_rope_bf16cvt(void);    // proj_q, proj_k
cute_fusion_fn cute_fuse_dequant_bf16cvt(void);         // proj_v
cute_fusion_fn cute_fuse_masked_softmax_kvscale_bf16cvt(void);  // scores
cute_fusion_fn cute_fuse_dequant_silu(void);             // ffn_gate
cute_fusion_fn cute_fuse_dequant_hadamard(void);         // ffn_up
cute_fusion_fn cute_fuse_dequant_resadd(void);           // proj_o, ffn_down

// 直接调用版本（非回调，独立使用）
void cute_fuse_dequant_rope_bf16cvt_direct(...);
void cute_fuse_dequant_bf16cvt_direct(...);
// ... etc
```

实际上，由于 fusion 函数的参数不完全统一（有的需要 pos，有的需要 output_scale），`cute_fusion_fn` 的 `void *ctx` 需要承载不同上下文。建议定义：

```c
// RoPE fusion 的上下文
typedef struct {
    int pos;   // 当前 sequence position offset
} cute_rope_ctx_t;

// Masked softmax 的上下文
typedef struct {
    int pos;
    void *bitmask;
    int max_ctx;
} cute_softmax_ctx_t;

// Hadamard 的上下文
typedef struct {
    void *output_max;  // absmax 输出
} cute_hadamard_ctx_t;
```

#### C.3 独立操作

```c
// cute_quant.h

// F32 → INT8 per-token 量化
void cute_smoothquant(float *input, int dim_i, int dim_j,
                      int8_t *output, float *output_scale,
                      int need_stage1);

// RMS 归一化
void cute_rmsnorm(float *input, float *output,
                  float *weight, float rms_epsilon,
                  int batch, int seq_len, int hidden_dim);

// RMS 归一化 + 提取 per-token absmax
void cute_rmsnorm_with_scale(float *input, float *output,
                             float *weight, float *per_token_scale,
                             float rms_epsilon,
                             int batch, int seq_len, int hidden_dim);
```

#### C.4 底层向量数学（internal）

`cute_vec_math.h` 内部使用，不暴露给上层：

```c
// 来自 llama3_1B.c 的向量化实现，全部保留
vfloat32m4_t vec_exp(vfloat32m4_t x, size_t vl);
vfloat32m4_t vec_sin(vfloat32m4_t x, size_t vl);
vfloat32m4_t vec_cos(vfloat32m4_t x, size_t vl);
```

#### C.5 CMake

```cmake
# L3: cutelib/fusion — CPU post-processing（header-only，依赖 cutelib_tensor）
add_library(cutelib_fusion INTERFACE)
target_include_directories(cutelib_fusion INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/fusion/include
)
target_link_libraries(cutelib_fusion INTERFACE cutelib_tensor)
```

#### C.6 测试

对每个 fusion 函数独立验证：

| Test | 输入 | 验证 |
|---|---|---|
| `test_fuse_dequant_bf16cvt` | 64×64 I32 + scale | 输出与 llama3_1B.c 的函数 bit-exact |
| `test_fuse_dequant_silu` | 64×64 I32 + scale | bit-exact |
| `test_smoothquant` | 128×2048 F32 | 输出 INT8 + scale 与原实现一致 |
| `test_rmsnorm` | 128×2048 F32 + weight | bit-exact |

golden 数据来源：从 llama3_1B.c 中提取固定输入，截取中间结果作为 golden。

### Phase D：L4 cutelib/model

> 目标：用 L1-L3 的 API 编排 llama_block。

#### D.1 目录结构

```text
cutelib/model/
└── include/
    └── cute_llama.h
```

#### D.2 cute_llama_block 实现

```c
// cute_llama.h

typedef struct {
    int seq_len;
    int embed_dim;
    int key_dim;
    int value_dim;
    int n_head_q;
    int n_head_kv;
    int ffn_dim;
    float rms_epsilon;
    float inv_sqrt_key_dim;

    // 权重（外部提供指针）
    float    *attn_norm_weight;
    int8_t   *proj_q_weight;  float *proj_q_scale;
    int8_t   *proj_k_weight;  float *proj_k_scale;
    int8_t   *proj_v_weight;  float *proj_v_scale;
    int8_t   *proj_o_weight;  float *proj_o_scale;
    float    *ffn_norm_weight;
    int8_t   *ffn_gate_weight; float *ffn_gate_scale;
    int8_t   *ffn_up_weight;   float *ffn_up_scale;
    int8_t   *ffn_down_weight; float *ffn_down_scale;

    // RoPE
    float    *rope_theta;

    // 工作缓冲区（外部提供）
    void     *double_buf[4];  // pipeline 缓冲，每个 >= max tile
    void     *tcm_buf;        // 中间结果缓冲
    int8_t   *quant_buf;      // 量化工作 buffer
    float    *quant_scale_buf;
    // ... 其他中间 buffer
} cute_llama_config_t;

void cute_llama_block(cute_llama_config_t *cfg, float *input, float *output);
```

`cute_llama_block` 的内部就是 [3.4 节](#34-llama_block-调用序列) 的 13 步调用序列，每一步调用 `cute_tiled_matmul` 或 `cute_smoothquant` / `cute_rmsnorm`。

#### D.3 CMake

```cmake
# L4: cutelib/model — model-level（header-only，依赖 cutelib_fusion）
add_library(cutelib_model INTERFACE)
target_include_directories(cutelib_model INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/model/include
)
target_link_libraries(cutelib_model INTERFACE cutelib_fusion)
```

#### D.4 端到端验证

```text
test_llama_block:
  1. 准备输入 golden（从 llama3_1B.c 的 identity 数组导入）
  2. 准备权重 golden（从 llama3_1B.c 的 weight 数组导入）
  3. 调用 cute_llama_block()
  4. 比对 output 与 llama3_1B.c 的输出
  5. memverify bit-exact
```

---

## 5. 依赖图与执行顺序

```text
Phase A (L1 runtime 扩展)
    │
    ▼
Phase B (L2 tensor)
    │
    ├──────► Phase B 测试 (tensor correctness)
    │
    ▼
Phase C (L3 fusion)
    │
    ├──────► Phase C 测试 (fusion correctness)
    │
    ▼
Phase D (L4 model)
    │
    └──────► Phase D 端到端测试 (llama_block bit-exact)
```

每个 Phase 内部可以并行开发（例如 B.1 tensor descriptor 和 C.2 fusion 函数无依赖关系），但测试必须按顺序。

---

## 6. 文件清单

### Phase A: runtime 扩展

| 操作 | 文件 |
|------|------|
| 修改 | `cutelib/runtime/cute_runtime.h` |

### Phase B: cutelib/tensor

| 操作 | 文件 |
|------|------|
| 创建 | `cutelib/tensor/include/cute_tensor.h` |
| 创建 | `cutelib/tensor/include/cute_ops.h` |
| 创建 | `tests/tensor/matmul/tensor_matmul_i8_128_128_128_zeroinit/` |
| 创建 | `tests/tensor/matmul/tensor_matmul_i8_256_256_256_zeroinit/` |
| 修改 | `CMakeLists.txt` |

### Phase C: cutelib/fusion

| 操作 | 文件 |
|------|------|
| 创建 | `cutelib/fusion/include/cute_fusion.h` |
| 创建 | `cutelib/fusion/include/cute_quant.h` |
| 创建 | `cutelib/fusion/include/cute_vec_math.h` |
| 创建 | `tests/fusion/test_fuse_dequant_bf16cvt.c` |
| 创建 | `tests/fusion/test_fuse_dequant_silu.c` |
| 创建 | `tests/fusion/test_smoothquant.c` |
| 创建 | `tests/fusion/test_rmsnorm.c` |
| 修改 | `CMakeLists.txt` |

### Phase D: cutelib/model

| 操作 | 文件 |
|------|------|
| 创建 | `cutelib/model/include/cute_llama.h` |
| 创建 | `tests/model/test_llama_block.c` |
| 修改 | `CMakeLists.txt` |

---

## 7. 风险与待定

| 风险 | 缓解 |
|------|------|
| `cute_tiled_matmul` 的 softmax 分支 tiling 模式不同（只按 M tile） | 第一版先不支持 softmax fusion 的 tiled 路径，softmax matmul 用循环单 tile + 独立 softmax 调用 |
| fusion 函数依赖 RISC-V V 扩展，golden 数据生成需要硬件或模拟器 | 用 cutetest 中已有的真实数据作为 golden |
| 四缓冲区大小需要 >= 最大 tile 输出（当前最大是 FFN_DIM=8192 的 tile） | 缓冲区统一分配为 `CUTE_TILE_M × max_N × 4` byte |
| llama3_1B.c 中部分 buffer 是 alias（如 `ffn_gate_buf_f32 == ffn_up_buf_f32`） | model 层显式管理 buffer alias，或在 config 中用 union |
