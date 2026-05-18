# Phase C1 / Phase 3b: vector primitive 接口说明

## Context

这份文档收束 `llama3_1B.c` 里所有真正的向量任务接口。它们都属于 `cutelib/primitive`，不是 fusion case，也不谈 `notile` / `nopipeline` / `pipeline`。

Phase C1 的职责是把这些 helper 固定成稳定 API，让后续 fusion case 只负责把 matmul 输出喂给它们。数学实现和 bit-exact 验证仍以前置的 [`phaseC0_single_primitive.md`](phaseC0_single_primitive.md) 为准。

---

## 1. 向量闭包

| primitive | `llama3_1B.c` 里的角色 | 备注 |
|---|---|---|
| `cute_fast_sqrt` | `RMSnorm` / `RMSnorm_With_getabsmax_scale` | shared scalar helper |
| `cute_vec_exp` | SiLU / softmax | RVV exp helper |
| `cute_vec_sin` | RoPE | RVV sin helper |
| `cute_vec_cos` | RoPE | RVV cos helper |
| `cute_dequant_i32_to_f32_tile` | `fuse_ops_DEQUANT_ROPE_BF16CVRT`、`fuse_ops_DEQUANT_SILU`、`fuse_ops_DEQUANT_HADAMARD`、`fuse_ops_DEQUANT_RESADD` | accumulator dequant |
| `cute_dequant_i32_to_bf16_tile` | `fuse_ops_DEQUANT_BF16CVRT_With_T` | layout-aware dequant/store adapter |
| `cute_f32_to_f16_tile` / `cute_f32_to_bf16_tile` | RoPE / softmax 输出 store | 当前代码里用 `_Float16` 路径保持 bit-exact |
| `cute_silu_tile` | `fuse_ops_DEQUANT_SILU` | elementwise SiLU |
| `cute_hadamard_tile` | `fuse_ops_DEQUANT_HADAMARD` | elementwise multiply + row absmax |
| `cute_resadd_tile` | `fuse_ops_DEQUANT_RESADD` | residual add |
| `cute_rope_bf16_tile` | `fuse_ops_DEQUANT_ROPE_BF16CVRT` | RoPE 后处理 |
| `cute_masked_softmax_bf16_tile` | `softmax_cvrtfp16` / `fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT` | causal masked softmax |
| `cute_smoothquant` | `smoothquant` | quant stage 1/2 合并接口 |
| `cute_rmsnorm` | `RMSnorm` | standalone norm |
| `cute_rmsnorm_with_scale` | `RMSnorm_With_getabsmax_scale` | norm + absmax scale |

`cute_silu_scalar` 和 `cute_smoothquant_stage1_getscale` / `cute_smoothquant_stage2_quant` 是内部实现细节，不单独暴露给 fusion case。

---

## 2. 接口边界

primitive 层只接收普通 buffer、stride、shape 和显式参数：

- 不接收 `cute_post_call_t`。
- 不知道 CUTE task id。
- 不管理 double buffer。
- 不区分 `notile` / `nopipeline` / `pipeline`。
- 不读 `llama3_1B.c` 的全局变量。

换句话说，primitive 的接口是“向量任务本身”，fusion case 的接口才是“matmul 输出如何喂给向量任务”。

---

## 3. 与 `llama3_1B.c` 对照

| `llama3_1B.c` 项 | 对应 primitive | 备注 |
|---|---|---|
| `fast_sqrt` | `cute_fast_sqrt` | 被 RMSNorm 共享 |
| `vec_exp` | `cute_vec_exp` | 被 SiLU / softmax 共享 |
| `vec_sin` / `vec_cos` | `cute_vec_sin` / `cute_vec_cos` | RoPE 共享 |
| `fuse_ops_DEQUANT_ROPE_BF16CVRT` | `cute_dequant_i32_to_f32_tile` + `cute_rope_bf16_tile` | 由 fusion case 负责串接 |
| `fuse_ops_DEQUANT_BF16CVRT_With_T` | `cute_dequant_i32_to_bf16_tile` + layout adapter | 这是唯一需要特别注意布局的 dequant 路径 |
| `softmax_cvrtfp16` / `fuse_ops_MASKED_SOFTMAX_KVSCALE_BF16CVRT` | `cute_masked_softmax_bf16_tile` | causal mask + kv scale |
| `fuse_ops_DEQUANT_SILU` | `cute_dequant_i32_to_f32_tile` + `cute_silu_tile` | 纯向量后处理 |
| `fuse_ops_DEQUANT_HADAMARD` | `cute_dequant_i32_to_f32_tile` + `cute_hadamard_tile` | 需要 row absmax 累积 |
| `fuse_ops_DEQUANT_RESADD` | `cute_dequant_i32_to_f32_tile` + `cute_resadd_tile` | 纯向量后处理 |
| `smoothquant` | `cute_smoothquant` | 两阶段接口收敛为一个 public API |
| `RMSnorm` | `cute_rmsnorm` | 共享 `cute_fast_sqrt` |
| `RMSnorm_With_getabsmax_scale` | `cute_rmsnorm_with_scale` | 共享 `cute_fast_sqrt` + absmax |

`silu()` 本体在 `llama3_1B.c` 里是空壳，不是独立数学接口；真正的逻辑在 `fuse_ops_DEQUANT_SILU` 里。

---

## 4. 验证边界

每个 primitive 使用独立 golden 和独立测试：

| 测试 | 覆盖 |
|---|---|
| `primitive_vec_math_n256` | `cute_vec_exp` / `cute_vec_sin` / `cute_vec_cos` |
| `primitive_dequant_f32_m64_n64` | `cute_dequant_i32_to_f32_tile` |
| `primitive_dequant_f16_m64_n64` | `cute_dequant_i32_to_f16_tile` / `cute_dequant_i32_to_bf16_tile` |
| `primitive_silu_m128_n128` | `cute_silu_tile` |
| `primitive_resadd_m64_n64` | `cute_resadd_tile` |
| `primitive_hadamard_m64_n128` | `cute_hadamard_tile` + `row_absmax` 跨 tile 累积 |
| `primitive_rope_pos0_m64_head_dim64_n_head1` | `cute_rope_bf16_tile` |
| `primitive_rope_pos17_m64_head_dim64_n_head1` | `cute_rope_bf16_tile` |
| `primitive_masked_softmax_m64_n128` | `cute_masked_softmax_bf16_tile` |
| `primitive_smoothquant_m128_k2048` | `cute_smoothquant` |
| `primitive_rmsnorm_batch1_seq_len128_hidden_dim2048` | `cute_rmsnorm` + `cute_fast_sqrt` |
| `primitive_rmsnorm_scale_batch1_seq_len128_hidden_dim2048` | `cute_rmsnorm_with_scale` + `cute_fast_sqrt` + absmax |

`cute_fast_sqrt` 没有单独的 primitive case，但已经被两组 RMSNorm 测试直接覆盖。

---

## 5. Done Criteria

1. `llama3_1B.c` 里使用到的所有向量 helper 都能在 `cutelib/primitive/include/` 找到对应 public API。
2. 每个 primitive 有独立测试和固定 golden。
3. primitive API 不依赖 `cute_tiled_matmul`、`cute_post_call_t` 或 pipeline buffer。
4. fusion case 只做参数适配和调度，不重复实现向量数学逻辑。
