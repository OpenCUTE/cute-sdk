#ifndef CUTE_TEST_LLAMA_STAGE_NONZERO_TEMPLATE_H
#define CUTE_TEST_LLAMA_STAGE_NONZERO_TEMPLATE_H

#include <stddef.h>
#include <stdint.h>

#include "cute_llama.h"
#include "golden/manual/layer/llama_block_nonzero_weight_1b_shape_seq128/golden_llama_block_inputs.h"
#include "tests/primitive/primitive_test_utils.h"

#define TEST_SEQ_LEN GOLDEN_LLAMA_SEQ_LEN
#define TEST_EMBED_DIM GOLDEN_LLAMA_EMBED_DIM
#define TEST_KEY_DIM GOLDEN_LLAMA_KEY_DIM
#define TEST_VALUE_DIM GOLDEN_LLAMA_VALUE_DIM
#define TEST_N_HEAD_Q GOLDEN_LLAMA_N_HEAD_Q
#define TEST_N_HEAD_KV GOLDEN_LLAMA_N_HEAD_KV
#define TEST_FFN_DIM GOLDEN_LLAMA_FFN_DIM
#define TEST_MAX_CTX_LEN GOLDEN_LLAMA_MAX_CTX_LEN

#define TEST_Q_DIM (TEST_N_HEAD_Q * TEST_KEY_DIM)
#define TEST_KV_DIM (TEST_N_HEAD_KV * TEST_KEY_DIM)
#define TEST_V_DIM (TEST_N_HEAD_KV * TEST_VALUE_DIM)

#define CUTE_STAGE_OUTPUT_ALIGN __attribute__((used, aligned(256)))

static float scratch[2][CUTE_TILE_M][TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static uint32_t zero_bias_placeholder CUTE_TEST_ALIGN;

static inline cute_llama_block_config_t stage_base_config(void)
{
    cute_llama_block_config_t cfg = {
        .seq_len = TEST_SEQ_LEN,
        .embed_dim = TEST_EMBED_DIM,
        .key_dim = TEST_KEY_DIM,
        .value_dim = TEST_VALUE_DIM,
        .n_head_q = TEST_N_HEAD_Q,
        .n_head_kv = TEST_N_HEAD_KV,
        .ffn_dim = TEST_FFN_DIM,
        .max_ctx_len = TEST_MAX_CTX_LEN,
        .rms_epsilon = 1.0e-5f,
        .kv_scale = 0.125f,
        .use_pipeline = 1,
        .attn_norm_weight = golden_llama_attn_norm_weight,
        .ffn_norm_weight = golden_llama_ffn_norm_weight,
        .rope_theta = golden_llama_rope_theta,
        .causal_mask = golden_llama_causal_mask,
    };
    return cfg;
}

static inline cute_llama_block_workspace_t stage_workspace(uint32_t *zero_bias)
{
    cute_llama_block_workspace_t ws = {
        .scratch0 = scratch[0],
        .scratch1 = scratch[1],
        .zero_bias = zero_bias,
    };
    return ws;
}

static inline void stage_scale_f32(float *data, int count, float scale)
{
    for (int i = 0; i < count; ++i) {
        data[i] *= scale;
    }
}

#if defined(LLAMA_STAGE_PROJ_Q)

static uint16_t output[TEST_SEQ_LEN][TEST_Q_DIM] CUTE_STAGE_OUTPUT_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();
    cfg.proj_q_weight = golden_llama_proj_q_weight;
    cfg.proj_q_weight_scale = golden_llama_proj_q_scale;

    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t a = {(void *)golden_llama_attn_norm_q8,
                       (uint64_t)TEST_EMBED_DIM,
                       TEST_SEQ_LEN, TEST_EMBED_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t weight = {(void *)cfg.proj_q_weight,
                            (uint64_t)TEST_EMBED_DIM,
                            TEST_EMBED_DIM, TEST_Q_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_Q_DIM, CUTEDataTypeI8I8I32};
    cute_rope_ctx_t rope_ctx = {
        .pos = 0,
        .rope_theta = cfg.rope_theta,
        .key_dim = TEST_KEY_DIM,
    };

    cute_llama_matmul_post(&cfg, &a, &weight, output,
                           (uint64_t)TEST_Q_DIM * sizeof(output[0][0]),
                           sizeof(output[0][0]), &bias,
                           (float *)golden_llama_attn_norm_scale,
                           (float *)cfg.proj_q_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, &ws,
                           cute_post_dequant_rope_bf16cvt, &rope_ctx);
    cute_test_host_verify_barrier();
    return 0;
}

#elif defined(LLAMA_STAGE_PROJ_K)

static uint16_t output[TEST_SEQ_LEN][TEST_KV_DIM] CUTE_STAGE_OUTPUT_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();
    cfg.proj_k_weight = golden_llama_proj_k_weight;
    cfg.proj_k_weight_scale = golden_llama_proj_k_scale;

    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t a = {(void *)golden_llama_attn_norm_q8,
                       (uint64_t)TEST_EMBED_DIM,
                       TEST_SEQ_LEN, TEST_EMBED_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t weight = {(void *)cfg.proj_k_weight,
                            (uint64_t)TEST_EMBED_DIM,
                            TEST_EMBED_DIM, TEST_KV_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_KV_DIM, CUTEDataTypeI8I8I32};
    cute_rope_ctx_t rope_ctx = {
        .pos = 0,
        .rope_theta = cfg.rope_theta,
        .key_dim = TEST_KEY_DIM,
    };

    cute_llama_matmul_post(&cfg, &a, &weight, output,
                           (uint64_t)TEST_KV_DIM * sizeof(output[0][0]),
                           sizeof(output[0][0]), &bias,
                           (float *)golden_llama_attn_norm_scale,
                           (float *)cfg.proj_k_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, &ws,
                           cute_post_dequant_rope_bf16cvt, &rope_ctx);
    cute_test_host_verify_barrier();
    return 0;
}

#elif defined(LLAMA_STAGE_PROJ_V)

static uint16_t output[TEST_V_DIM][TEST_SEQ_LEN] CUTE_STAGE_OUTPUT_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();
    cfg.proj_v_weight = golden_llama_proj_v_weight;
    cfg.proj_v_weight_scale = golden_llama_proj_v_scale;

    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t a = {(void *)golden_llama_attn_norm_q8,
                       (uint64_t)TEST_EMBED_DIM,
                       TEST_SEQ_LEN, TEST_EMBED_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t weight = {(void *)cfg.proj_v_weight,
                            (uint64_t)TEST_EMBED_DIM,
                            TEST_EMBED_DIM, TEST_V_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_V_DIM, CUTEDataTypeI8I8I32};

    cute_llama_matmul_post(&cfg, &a, &weight, output,
                           (uint64_t)TEST_SEQ_LEN * sizeof(output[0][0]),
                           sizeof(output[0][0]), &bias,
                           (float *)golden_llama_attn_norm_scale,
                           (float *)cfg.proj_v_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 1, &ws,
                           cute_post_dequant_bf16cvt, NULL);
    cute_test_host_verify_barrier();
    return 0;
}

#elif defined(LLAMA_STAGE_SCORE_HEAD0)

static uint16_t output[TEST_SEQ_LEN][TEST_SEQ_LEN] CUTE_STAGE_OUTPUT_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();

    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t q = {(void *)golden_llama_q_f16,
                       (uint64_t)TEST_Q_DIM * sizeof(uint16_t),
                       TEST_SEQ_LEN, TEST_KEY_DIM, CUTEDataTypeBF16BF16F32};
    cute_tensor_t k = {(void *)golden_llama_k_f16,
                       (uint64_t)TEST_KV_DIM * sizeof(uint16_t),
                       TEST_KEY_DIM, TEST_SEQ_LEN, CUTEDataTypeBF16BF16F32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_SEQ_LEN,
                          CUTEDataTypeBF16BF16F32};
    cute_softmax_ctx_t softmax_ctx = {
        .pos = 0,
        .bitmask = cfg.causal_mask,
        .max_ctx_len = cfg.max_ctx_len,
        .kv_scale = cfg.kv_scale,
    };

    cute_llama_matmul_row_post(&cfg, &q, &k, output,
                               (uint64_t)TEST_SEQ_LEN * sizeof(output[0][0]),
                               sizeof(output[0][0]), &bias, &ws,
                               cute_post_masked_softmax_kvscale_bf16cvt,
                               &softmax_ctx);
    cute_test_host_verify_barrier();
    return 0;
}

#elif defined(LLAMA_STAGE_FFN_GATE)

static float output[TEST_SEQ_LEN][TEST_FFN_DIM] CUTE_STAGE_OUTPUT_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();
    cfg.ffn_gate_weight = golden_llama_ffn_gate_weight;
    cfg.ffn_gate_weight_scale = golden_llama_ffn_gate_scale;

    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t a = {(void *)golden_llama_ffn_norm_q8,
                       (uint64_t)TEST_EMBED_DIM,
                       TEST_SEQ_LEN, TEST_EMBED_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t weight = {(void *)cfg.ffn_gate_weight,
                            (uint64_t)TEST_EMBED_DIM,
                            TEST_EMBED_DIM, TEST_FFN_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_FFN_DIM,
                          CUTEDataTypeI8I8I32};

    cute_llama_matmul_post(&cfg, &a, &weight, output,
                           (uint64_t)TEST_FFN_DIM * sizeof(output[0][0]),
                           sizeof(output[0][0]), &bias,
                           (float *)golden_llama_ffn_norm_scale,
                           (float *)cfg.ffn_gate_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, &ws,
                           cute_post_dequant_silu, NULL);
    cute_test_host_verify_barrier();
    return 0;
}

#elif defined(LLAMA_STAGE_FFN_UP)

static float output[TEST_SEQ_LEN][TEST_FFN_DIM] CUTE_STAGE_OUTPUT_ALIGN;
static float row_absmax[TEST_SEQ_LEN] CUTE_TEST_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();
    cfg.ffn_up_weight = golden_llama_ffn_up_weight;
    cfg.ffn_up_weight_scale = golden_llama_ffn_up_scale;

    cute_llama_zero_f32(row_absmax, TEST_SEQ_LEN);
    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t a = {(void *)golden_llama_ffn_norm_q8,
                       (uint64_t)TEST_EMBED_DIM,
                       TEST_SEQ_LEN, TEST_EMBED_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t weight = {(void *)cfg.ffn_up_weight,
                            (uint64_t)TEST_EMBED_DIM,
                            TEST_EMBED_DIM, TEST_FFN_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_FFN_DIM,
                          CUTEDataTypeI8I8I32};
    cute_hadamard_ctx_t hadamard_ctx = {
        .lhs = golden_llama_ffn_gate_f32,
        .lhs_stride = (uint64_t)TEST_FFN_DIM * sizeof(float),
        .output_absmax = row_absmax,
    };

    cute_llama_matmul_post(&cfg, &a, &weight, output,
                           (uint64_t)TEST_FFN_DIM * sizeof(output[0][0]),
                           sizeof(output[0][0]), &bias,
                           (float *)golden_llama_ffn_norm_scale,
                           (float *)cfg.ffn_up_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, &ws,
                           cute_post_dequant_hadamard, &hadamard_ctx);
    stage_scale_f32(row_absmax, TEST_SEQ_LEN, 1.0f / 127.0f);
    cute_test_publish_f32_trace(row_absmax, TEST_SEQ_LEN);
    cute_test_host_verify_barrier();
    return 0;
}

#elif defined(LLAMA_STAGE_FFN_DOWN)

static float output[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_STAGE_OUTPUT_ALIGN;

int main(void)
{
    cute_llama_block_config_t cfg = stage_base_config();
    cfg.ffn_down_weight = golden_llama_ffn_down_weight;
    cfg.ffn_down_weight_scale = golden_llama_ffn_down_scale;

    cute_llama_block_workspace_t ws = stage_workspace(&zero_bias_placeholder);
    cute_tensor_t a = {(void *)golden_llama_ffn_up_q8,
                       (uint64_t)TEST_FFN_DIM,
                       TEST_SEQ_LEN, TEST_FFN_DIM, CUTEDataTypeI8I8I32};
    cute_tensor_t weight = {(void *)cfg.ffn_down_weight,
                            (uint64_t)TEST_FFN_DIM,
                            TEST_FFN_DIM, TEST_EMBED_DIM,
                            CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, TEST_SEQ_LEN, TEST_EMBED_DIM,
                          CUTEDataTypeI8I8I32};
    cute_resadd_ctx_t resadd_ctx = {
        .residual = golden_llama_proj_o_f32,
        .residual_stride = (uint64_t)TEST_EMBED_DIM * sizeof(float),
    };

    cute_llama_matmul_post(&cfg, &a, &weight, output,
                           (uint64_t)TEST_EMBED_DIM * sizeof(output[0][0]),
                           sizeof(output[0][0]), &bias,
                           (float *)golden_llama_ffn_up_row_scale,
                           (float *)cfg.ffn_down_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, &ws,
                           cute_post_dequant_resadd, &resadd_ctx);
    cute_test_host_verify_barrier();
    return 0;
}

#else
#error "select one LLAMA_STAGE_* macro before including this header"
#endif

#endif /* CUTE_TEST_LLAMA_STAGE_NONZERO_TEMPLATE_H */
