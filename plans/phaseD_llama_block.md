# Phase D: LLaMA block 收敛计划

## Context

目标是用 `cute-sdk` 分层 API 重写并验证
`cutetest/transformer_test/llama/llama3_1B.c` 里的单层
`llama_block`。

当前底层能力已经基本闭环：

- `cutelib/runtime`: `cute_matmul` / `cute_blockscale_matmul` / task wait 可用。
- `cutelib/tensor`: tiled matmul + `notile` / `nopipeline` / `pipeline` post-op 可用。
- `cutelib/primitive`: RMSNorm / SmoothQuant / dequant / RoPE / masked softmax / SiLU / Hadamard / ResAdd 可用。
- `cutelib/fusion`: 6 个 vector fusion post-op 已接入 tensor matmul。
- `tests/fusion.yaml`: 6 个 fusion case、18 个 binary 已通过。

剩下的重点不是补普通 vector primitive，而是把 LLaMA 真实执行路径里的
attention / layer 级语义补齐。

---

## 1. LLaMA block 算子路径

`llama3_1B.c` 当前配置：

| 参数 | 值 |
|---|---:|
| `SEQ_LEN` | 128 |
| `EMBEDING_DIMENSION` | 2048 |
| `KEY_DIMENSION` | 64 |
| `VALUE_DIMENSION` | 64 |
| `N_HEAD_Q` | 32 |
| `N_HEAD_KV` | 8 |
| `FFN_DIMENSION` | 8192 |
| `INV_SQRT_KEY_DIMENSION` | 0.125 |

单层调用序列：

```text
1.  RMSNorm + scale
2.  SmoothQuant
3.  Q matmul + dequant + RoPE + F16/BF16 store
4.  K matmul + dequant + RoPE + F16/BF16 store
5.  V matmul + dequant + F16/BF16 store
6.  QK scores matmul + masked softmax + kv scale + F16/BF16 store
7.  softmax(scores) x V matmul -> attention context F32
8.  SmoothQuant(attention context)
9.  O projection matmul + dequant + residual add
10. RMSNorm + scale
11. SmoothQuant
12. FFN gate matmul + dequant + SiLU
13. FFN up matmul + dequant + Hadamard + row absmax
14. SmoothQuant(FFN hidden)
15. FFN down matmul + dequant + residual add
```

---

## 2. 已覆盖和缺口

| 路径 | 当前状态 | 缺口 |
|---|---|---|
| RMSNorm / RMSNorm with scale | primitive 已测 | 需要 layer 级真实 buffer 接线 |
| SmoothQuant stage1=true | primitive 已测 | 需要真实 shape 回归 |
| SmoothQuant stage1=false | API 已有 | 需要补独立 case |
| Q/K projection + RoPE | fusion 已测 `N=64` | 需要真实 shape/head layout 回归 |
| V projection + bf16cvt | fusion 已测 | 需要真实 shape/head layout 回归 |
| QK scores + masked softmax | fusion 已测 `N=64` | **缺 `N=128` 整行 softmax / row-block fusion** |
| softmax x V context matmul | tensor 能力疑似已有 | **缺 attention context 专项 case** |
| O projection + resadd | fusion 已测小 shape | 需要真实 shape回归 |
| FFN gate + SiLU | fusion 已测小 shape | 需要 `N=8192` 回归 |
| FFN up + Hadamard + absmax | fusion 已测小 shape | 需要 `N=8192` 多 tile absmax 回归 |
| FFN down + resadd | fusion 已测小 shape | 需要 `K=8192` 回归 |
| layer `cute_llama_block` | 未实现 | **缺 L4 layer API 与端到端 case** |

---

## 3. 新增计划总览

建议拆成四个小阶段：

| 阶段 | 目标 | 产物 |
|---|---|---|
| D0 | 补 SmoothQuant false 和 attention context 基础 case | primitive/tensor tests |
| D1 | 补 `N=128` masked softmax tensor fusion | fusion case + helper |
| D2 | 补 LLaMA 真实 shape fusion 回归 | llama-shape fusion suite |
| D3 | 实现 `cute_llama_block` layer API | `cutelib/layer` + `tests/layer` |

执行顺序必须先 D1，再 D3。因为 layer attention 依赖整行 softmax 语义。

---

## 4. D0: 前置补强 case

### 4.1 SmoothQuant `need_stage1=false`

原因：

- LLaMA 中 RMSNorm with scale 先生成 per-token scale。
- 随后调用 `smoothquant(..., need_stage1=false)` 只做 stage2 quant。
- 当前 `primitive_smoothquant_m128_k2048` 只覆盖 `need_stage1=true`。

新增：

```text
tests/primitive/primitive_smoothquant_stage2_m128_k2048/
    case.json
    test.c
golden/manual/vector/smoothquant_stage2_m128_k2048/
    manifest.json
    golden_output.bin
    golden_scale.bin
```

测试策略：

- 输入复用现有 `smoothquant_m128_k2048` 的 F32 input。
- scale 直接使用 golden scale。
- 调 `cute_smoothquant(input, M, K, output, scale, false)`。
- verify:
  - `output`: bit exact
  - `scale`: 可选，只作为 trace publish 确认未被破坏。

### 4.2 Attention context matmul

对应 LLaMA 步骤：

```text
scores_q16[SEQ_LEN, SEQ_LEN] x v_q16[SEQ_LEN, VALUE_DIMENSION]
  -> context_f32[SEQ_LEN, VALUE_DIMENSION]

M=128, N=64, K=128
dtype=CUTEDataTypeF16F16F32
after_ops=NO_ACTIVATION
```

新增：

```text
tests/fusion/fusion_attention_context_f16_m128_n64_k128/
    case.json
    test_notile.c
    test_nopipeline.c
    test_pipeline.c
golden/manual/fusion/attention_context_f16_m128_n64_k128/
    manifest.json
    golden_output.bin
```

验证重点：

- `F16F16F32` no-activation tiled matmul 的 stride/layout。
- `B` 按 CUTE 约定使用 `[N][K]` 存储，即每个 output column 一行 K。
- 输出是 F32，`float_tolerance_percent = 0.05`。

---

## 5. D1: `N=128` masked softmax tensor fusion

### 5.1 问题

当前 `fusion_matmul_masked_softmax_kvscale_bf16cvt` 使用：

```text
M=128, N=64, K=64
```

这能覆盖 vector primitive 和 row offset，但不能覆盖 LLaMA 真实 scores：

```text
M=128, N=128, K=64
```

softmax 必须对整行 128 个 score 归一化。如果用普通 `64x64` tile_j 拆成两个 tile，
每个 tile 独立 softmax，语义错误。

### 5.2 第一版策略：row-block full-N softmax

实现一个专用 tensor helper，只按 M 维切 tile，不按 N 维切 tile：

```c
cute_tiled_matmul_row_block_pipeline_ex(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    void *output,
    uint64_t output_stride,
    uint64_t output_elem_bytes,
    const cute_tensor_t *bias,
    int rows_per_tile,      // 64
    int full_cols,          // 128
    void *scratch0,
    void *scratch1,
    cute_post_op_fn post_op,
    void *post_ctx)
```

约束：

- `M % 64 == 0`
- `N` 是 softmax 完整行宽，第一版支持 `N=128`
- scratch 至少 `64 * N * sizeof(float)`
- post-op 的 `call.tile.cols = N`
- `call.tile.col0 = 0`
- `cute_post_masked_softmax_kvscale_bf16cvt` 继续复用，不复制数学逻辑。

### 5.3 新增 case

```text
tests/fusion/fusion_matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64/
    case.json
    test_notile.c
    test_nopipeline.c
    test_pipeline.c
golden/manual/fusion/matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64/
    manifest.json
    golden_output_f16.bin
```

输入：

- `A`: deterministic F16 `[128][64]`
- `B`: deterministic F16 `[128][64]`，按 CUTE B layout `[N][K]`
- `mask`: causal mask `[128][128]`
- `kv_scale = 0.125f`

验证：

- `golden_output_f16 -> output`
- F16/BF16 类输出 `float_tolerance_percent = 0.5`
- 如果 generator 和实现完全对齐，可接受 bit exact。

验收：

```bash
python3 tools/runner/cute-test.py \
  --suite cute-sdk/tests/fusion_attention.yaml \
  --skip-build
```

---

## 6. D2: LLaMA 真实 shape fusion 回归

D1 解决 attention softmax 语义后，需要把小 shape case 扩展到 LLaMA 真实维度。

### 6.1 Projection case

| case | shape | dtype | post-op |
|---|---|---|---|
| `fusion_llama_proj_q_rope` | M=128,N=2048,K=2048 | I8I8I32 | dequant+RoPE+F16 |
| `fusion_llama_proj_k_rope` | M=128,N=512,K=2048 | I8I8I32 | dequant+RoPE+F16 |
| `fusion_llama_proj_v_bf16cvt` | M=128,N=512,K=2048 | I8I8I32 | dequant+F16 |
| `fusion_llama_proj_o_resadd` | M=128,N=2048,K=2048 | I8I8I32 | dequant+resadd |

注意：

- Q has 32 heads，K/V has 8 KV heads。
- RoPE post-op 必须按每个 64-dim head 重置 `col0/head_dim` 语义。
- 当前 `fusion_matmul_dequant_rope_bf16cvt_m128_n64` 只覆盖一个完整 head。
- 真实 `N=2048/512` 时，需要确认 tensor post-op 按 tile_j 传入的 64 列刚好是一个 head，不跨 head。

### 6.2 FFN case

| case | shape | dtype | post-op |
|---|---|---|---|
| `fusion_llama_ffn_gate_silu` | M=128,N=8192,K=2048 | I8I8I32 | dequant+SiLU |
| `fusion_llama_ffn_up_hadamard` | M=128,N=8192,K=2048 | I8I8I32 | dequant+Hadamard+row_absmax |
| `fusion_llama_ffn_down_resadd` | M=128,N=2048,K=8192 | I8I8I32 | dequant+resadd |

验证重点：

- `N=8192` 下 Hadamard 的 row_absmax 跨 128 个 N tiles 正确累积。
- F32 输出 `float_tolerance_percent = 0.05`。
- Hadamard `row_absmax` 必须单独 verify。

### 6.3 Suite

新增：

```text
tests/llama_fusion.yaml
```

第一版可以只放较少大 shape smoke：

```yaml
cases:
  - fusion_llama_proj_q_rope
  - fusion_llama_attention_softmax
  - fusion_llama_attention_context
  - fusion_llama_ffn_up_hadamard
```

全量大 shape case 仿真会慢，默认不放入 `smoke.yaml`。

---

## 7. D3: `cutelib/layer` LLaMA block

### 7.1 新增目录

```text
cutelib/layer/include/
    cute_llama.h
tests/layer/
    llama_block_1b_seq128/
        case.json
        test.c
tests/llama_layer.yaml
```

CMake:

```cmake
add_library(cutelib_layer INTERFACE)
target_include_directories(cutelib_layer INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutelib/layer/include
)
target_link_libraries(cutelib_layer INTERFACE cutelib_fusion)
```

### 7.2 API 草案

```c
typedef struct {
    int seq_len;
    int embed_dim;
    int key_dim;
    int value_dim;
    int n_head_q;
    int n_head_kv;
    int ffn_dim;
    int max_ctx_len;
    float rms_epsilon;
    float kv_scale;

    const float *attn_norm_weight;
    const int8_t *proj_q_weight;
    const float *proj_q_weight_scale;
    const int8_t *proj_k_weight;
    const float *proj_k_weight_scale;
    const int8_t *proj_v_weight;
    const float *proj_v_weight_scale;
    const int8_t *proj_o_weight;
    const float *proj_o_weight_scale;

    const float *ffn_norm_weight;
    const int8_t *ffn_gate_weight;
    const float *ffn_gate_weight_scale;
    const int8_t *ffn_up_weight;
    const float *ffn_up_weight_scale;
    const int8_t *ffn_down_weight;
    const float *ffn_down_weight_scale;

    const float *rope_theta;
    const int8_t *causal_mask;

    void *scratch0;
    void *scratch1;
    void *scratch2;
    void *scratch3;
    void *workspace;
    size_t workspace_bytes;
} cute_llama_block_config_t;

void cute_llama_block(const cute_llama_block_config_t *cfg,
                      const float *input,
                      float *output);
```

### 7.3 Layer 内部调度

实现顺序严格对齐 `llama3_1B.c`：

```text
RMSNorm(input) + scale
SmoothQuant(false)
Q/K/V projection
for q_head:
  QK scores + full-row masked softmax
  softmax x V
SmoothQuant(attention context, true)
O projection + residual add
RMSNorm + scale
SmoothQuant(false)
FFN gate + SiLU
FFN up + Hadamard
SmoothQuant(false)
FFN down + residual add
```

### 7.4 Workspace 明确化

第一版不要隐藏分配。所有中间 buffer 由 test 或 caller 提供，避免裸机 malloc：

```text
hidden_q8                         [128][2048] I8
hidden_scale                      [128] F32
q_buf                             [128][32][64] F16
k_buf                             [128][8][64] F16
v_buf                             [128][8][64] F16
scores_buf                        [32][128][128] F16
attention_context                 [128][2048] F32
attention_q8                      [128][2048] I8
attention_scale                   [128] F32
proj_o_f32                        [128][2048] F32
proj_o_norm_f32                   [128][2048] F32
proj_o_norm_q8                    [128][2048] I8
ffn_gate_f32                      [128][8192] F32
ffn_up_f32                        [128][8192] F32
ffn_up_q8                         [128][8192] I8
ffn_up_scale                      [128] F32
```

后续再做 buffer alias / TCM reuse 优化。

---

## 8. 验证策略

### 8.1 分层验证

| 阶段 | 命令 | 目标 |
|---|---|---|
| D0 | `tests/vecprimitive.yaml` + new stage2 case | quant / context matmul 基础 |
| D1 | `tests/fusion_attention.yaml` | N=128 softmax 语义 |
| D2 | `tests/llama_fusion.yaml` | 真实 shape fusion |
| D3 | `tests/llama_layer.yaml` | 单层端到端 |

### 8.2 tolerance 规则

沿用当前约定：

- F32 输出：`float_tolerance_percent = 0.05`
- F16/BF16 storage 输出：`float_tolerance_percent = 0.5`
- I8/scale 等整数或稳定输出：优先 bit exact

### 8.3 Golden 来源

优先级：

1. 当前 `llama3_1B.c` 中间输出 dump。
2. `golden/manual/generators/*` deterministic generator。
3. 只有在无法确定硬件 layout 时，先补小 shape layout case，再扩展真实 shape。

---

## 9. Done Criteria

1. `fusion_matmul_masked_softmax_kvscale_bf16cvt_m128_n128_k64` 三 variant 通过。
2. `attention_context_f16_m128_n64_k128` 三 variant 通过。
3. SmoothQuant `need_stage1=false` 独立 case 通过。
4. 至少一组 LLaMA projection、attention、FFN 大 shape fusion case 通过。
5. `cutelib/layer/include/cute_llama.h` 提供 `cute_llama_block`。
6. `tests/layer/llama_block_1b_seq128` 可以完成单层 forward 并通过 memverify。

---

## 10. 风险

| 风险 | 处理 |
|---|---|
| CUTE matmul 是否支持 `N=128` 单 task 输出 | 先做 D1 小 case验证；不支持则改成 row-block 两阶段 softmax |
| full-row softmax scratch 太大 | 使用外部 scratch，第一版不自动分配 |
| RoPE 多 head tile_j 语义出错 | D2 单独做 `N=2048` projection 回归 |
| FFN 8192 仿真太慢 | 大 shape suite 不进入 smoke，先跑关键 subset |
| workspace alias 复杂 | D3 第一版显式 buffer，不做复用优化 |

