# Phase C0 / Phase 3a: 单算子 primitive 实现计划

## Context

Phase 3 应该先做 **单个算子 primitive**，不要一上来做 layer 编排，也不要先做 fused pipeline。

这里的单算子指：每个函数只完成一个清晰的数学/数据转换职责，可以独立构造输入、独立生成 golden、独立 bit-exact 验证。后续 Phase C1 再把这些 primitive 组合成 `cute_tiled_matmul` 的 fused post-op，并接入 pipeline。

---

## 1. 目标

Phase C0 产出一组可独立测试的基础算子：

1. matmul primitive：单次 `cute_matmul_op` / `cute_blockscale_matmul_op` 的稳定 API。
2. vec math primitive：`sqrt` / `exp` / `sin` / `cos` 等 RVV helper。
3. dtype/convert primitive：I32 dequant、F32 -> BF16、BF16/F16 store helper。
4. elementwise primitive：SiLU、Hadamard、Residual Add。
5. sequence primitive：RoPE、masked softmax。
6. quant/norm primitive：smoothquant、RMSNorm、RMSNorm + absmax scale。

Phase C0 不做 fused op，不做 pipeline overlap，不做 `llama_block`。

---

## 2. 文件结构

```text
cutelib/primitive/include/
    cute_vec_math.h       # RVV vec sqrt/exp/sin/cos helper
    cute_convert.h        # dequant / bf16 convert / typed load-store helper
    cute_elementwise.h    # silu / hadamard / residual add
    cute_sequence.h       # rope / masked softmax
    cute_quant.h          # smoothquant / rmsnorm

tests/primitive/
    test_vec_math/
    test_convert_dequant_bf16/
    test_elementwise_silu/
    test_elementwise_hadamard/
    test_elementwise_resadd/
    test_sequence_rope/
    test_sequence_masked_softmax/
    test_smoothquant/
    test_rmsnorm/
    test_rmsnorm_with_scale/
```

如果不想新增 `cutelib/primitive` 目录，也可以先放在 `cutelib/fusion/include/internal/`。但语义上 `primitive` 更准确：它不是 fusion，只是 fusion 依赖的单算子库。

---

## 3. Primitive API 草案

### 3.1 vec math

```c
// cute_vec_math.h
static inline float cute_fast_sqrt(float x);
static inline vfloat32m4_t cute_vec_exp(vfloat32m4_t x, size_t vl);
static inline vfloat32m4_t cute_vec_sin(vfloat32m4_t x, size_t vl);
static inline vfloat32m4_t cute_vec_cos(vfloat32m4_t x, size_t vl);
```

迁移原则：

- 从 `llama3_1B.c` 直接搬 `fast_sqrt`、`vec_exp`、`vec_sin`、`vec_cos`。
- 只改命名空间，不改常量、泰勒系数、RVV intrinsic 顺序。
- 不依赖任何 llama 全局变量。

### 3.2 convert / dequant

```c
// cute_convert.h
static inline float cute_dequant_i32(int32_t acc, float input_scale, float weight_scale);

void cute_dequant_i32_to_f32_tile(const int32_t *input, uint64_t input_stride,
                                  float *output, uint64_t output_stride,
                                  const float *input_scale,
                                  const float *weight_scale,
                                  int rows, int cols);

void cute_dequant_i32_to_bf16_tile(const int32_t *input, uint64_t input_stride,
                                   void *output, uint64_t output_stride,
                                   const float *input_scale,
                                   const float *weight_scale,
                                   int rows, int cols);
```

重点：

- scale 语义固定为 `input_scale[row] * weight_scale[0]`。
- 第一版只支持 per-token A + per-tensor B。
- BF16/F16 转换必须复用原实现里的转换方式，不能换近似路径。

### 3.3 elementwise

```c
// cute_elementwise.h
static inline float cute_silu_scalar(float x);

void cute_silu_tile(float *data, uint64_t stride, int rows, int cols);

void cute_hadamard_tile(const float *lhs, uint64_t lhs_stride,
                        const float *rhs, uint64_t rhs_stride,
                        float *output, uint64_t output_stride,
                        float *row_absmax,
                        int rows, int cols);

void cute_resadd_tile(const float *lhs, uint64_t lhs_stride,
                      const float *rhs, uint64_t rhs_stride,
                      float *output, uint64_t output_stride,
                      int rows, int cols);
```

重点：

- Hadamard 的 `row_absmax` 是 per row 累积输出，不能被后续 N tile 覆盖。
- ResAdd 只做 F32 + F32，不负责 dequant。
- SiLU 只对 F32 tile 做 elementwise，不负责 dequant。

### 3.4 sequence

```c
// cute_sequence.h
void cute_rope_bf16_tile(const float *input, uint64_t input_stride,
                         void *output, uint64_t output_stride,
                         const float *rope_theta,
                         int pos, int key_dim,
                         int rows, int cols);

void cute_masked_softmax_bf16_tile(const float *input, uint64_t input_stride,
                                   void *output, uint64_t output_stride,
                                   const int8_t *mask, uint64_t mask_stride,
                                   float scale,
                                   int row0, int col0,
                                   int rows, int cols);
```

重点：

- RoPE 只接收 F32 输入，输出 BF16；dequant 在 convert primitive 完成。
- Softmax 只接收 F32 scores，输出 BF16；matmul 和 mask 地址由上层传入。
- `row0/col0` 显式传入，避免 tile 内 softmax 错用局部行号。

### 3.5 quant / norm

```c
// cute_quant.h
void cute_smoothquant(float *input, int rows, int cols,
                      int8_t *output, float *output_scale,
                      bool need_stage1);

void cute_rmsnorm(float *input, float *output,
                  float *weight, float rms_epsilon,
                  int batch, int seq_len, int hidden_dim);

void cute_rmsnorm_with_scale(float *input, float *output,
                             float *weight, float *per_token_scale,
                             float rms_epsilon,
                             int batch, int seq_len, int hidden_dim);
```

这三个可以直接从 `llama3_1B.c` 迁出，先作为 standalone op 验证。

---

## 4. 从 llama3_1B.c 拆分关系

| 原 fusion 函数 | Phase C0 拆出的 primitive |
|---|---|
| `fuse_ops_DEQUANT_ROPE_BF16CVRT` | dequant_i32_to_f32 + rope_bf16 |
| `fuse_ops_DEQUANT_BF16CVRT_With_T` | dequant_i32_to_bf16 + layout store |
| `fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT` | masked_softmax_bf16 |
| `fuse_ops_DEQUANT_RESADD` | dequant_i32_to_f32 + resadd |
| `fuse_ops_DEQUANT_SILU` | dequant_i32_to_f32 + silu |
| `fuse_ops_DEQUANT_HADAMARD` | dequant_i32_to_f32 + hadamard + row_absmax |
| `smoothquant` | smoothquant |
| `RMSnorm` | rmsnorm |
| `RMSnorm_With_getabsmax_scale` | rmsnorm_with_scale |

Phase C1 的 fusion 函数只负责把这些 primitive 串起来，并适配 `cute_post_call_t`。

---

## 5. 测试计划

### 5.1 golden 来源

详细生成方案见 `plans/phaseC0_rvv_golden_generation.md`。Phase C0 golden
默认由 NVWA/llama reference RVV binary 在 `cuteqemu/build/qemu-riscv64` 上
执行后 dump `manifest.json + *.bin`。

优先级：

1. 使用 NVWA `llama3.2_1B/data_flow/gloden_opt.h` 和 standalone RVV
   primitive 生成 small golden。
2. 对 `dequant/resadd/hadamard` 等缺少 standalone op 的函数，从
   `llama3_1B.c` fusion 函数中拆出单 stage reference。
3. 使用 cutetest 里已有的中间结果头文件作为兼容性补充。

不要手写复杂 golden，尤其不要手写 softmax/RoPE。

### 5.2 测试矩阵

| 测试 | 输入 shape | 验证 |
|---|---:|---|
| `test_vec_math` | RVV vector lanes | exp/sin/cos 与原 helper bit-exact |
| `test_convert_dequant_bf16` | 64 x 64 I32 | F32/BF16 输出 bit-exact |
| `test_elementwise_silu` | 64 x 64 F32 | SiLU 输出 bit-exact |
| `test_elementwise_hadamard` | 64 x 128 F32 | output + row_absmax 跨 tile 累积 |
| `test_elementwise_resadd` | 64 x 64 F32 | residual add bit-exact |
| `test_sequence_rope` | 64 x 64 F32 | RoPE + BF16 bit-exact，覆盖 pos=0/pos>0 |
| `test_sequence_masked_softmax` | 64 x 128 F32 | causal mask + scale + BF16 bit-exact |
| `test_smoothquant` | 128 x 2048 F32 | INT8 + scale bit-exact |
| `test_rmsnorm` | 128 x 2048 F32 | norm 输出 bit-exact |
| `test_rmsnorm_with_scale` | 128 x 2048 F32 | norm 输出 + per-token scale bit-exact |

---

## 6. 执行顺序

| 步骤 | 内容 | 依赖 |
|------|------|------|
| 1 | 创建 `cutelib/primitive/include` 头文件骨架 | 无 |
| 2 | 迁移 vec math helper | 无 |
| 3 | 实现 dequant/BF16 convert primitive | vec math 无依赖 |
| 4 | 实现 SiLU/ResAdd/Hadamard primitive | convert 可独立 |
| 5 | 实现 RoPE primitive | vec sin/cos |
| 6 | 实现 masked softmax primitive | vec exp |
| 7 | 迁移 smoothquant/rmsnorm | vec sqrt |
| 8 | 添加 primitive 单测和 smoke 条目 | primitives |
| 9 | 进入 Phase C1 fusion/pipeline 组合 | primitive 单测通过 |

---

## 7. Done Criteria

Phase C0 完成时必须满足：

1. 所有 primitive 不依赖 `cute_tiled_matmul` pipeline。
2. 所有 primitive 不依赖 `llama_block` 全局 buffer。
3. 每个 primitive 都有独立测试和固定 golden。
4. dequant、RoPE、softmax、SiLU、Hadamard、ResAdd、smoothquant、RMSNorm 都能单独 bit-exact。
5. Phase C1 可以只写 glue/fusion adapter，不需要重新发明数学逻辑。

---

## 8. 不在 Phase C0 范围内

- 不实现 fused post-op。
- 不实现 pipeline overlap。
- 不实现 `cute_llama_block`。
- 不做 attention head/model layer 编排。
- 不优化算法，只迁移和拆分原实现。
