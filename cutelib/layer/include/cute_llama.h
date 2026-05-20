#ifndef CUTE_LLAMA_H
#define CUTE_LLAMA_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "cute_fusion.h"
#include "cute_quant.h"

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
    int use_pipeline;

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
} cute_llama_block_config_t;

typedef struct {
    float *attn_norm_f32;
    int8_t *attn_norm_q8;
    float *attn_norm_scale;

    uint16_t *q_bf16;
    uint16_t *k_bf16;
    uint16_t *v_bf16_t;
    uint16_t *scores_bf16;

    float *attn_context_f32;
    int8_t *attn_q8;
    float *attn_scale;
    float *proj_o_f32;

    float *ffn_norm_f32;
    int8_t *ffn_norm_q8;
    float *ffn_norm_scale;
    float *ffn_gate_f32;
    float *ffn_up_f32;
    int8_t *ffn_up_q8;
    float *ffn_up_scale;

    void *scratch0;
    void *scratch1;
    void *zero_bias;
} cute_llama_block_workspace_t;

static inline void cute_llama_zero_f32(float *data, int count)
{
    for (int i = 0; i < count; i++) {
        data[i] = 0.0f;
    }
}

static inline void cute_llama_matmul_post(
    const cute_llama_block_config_t *cfg,
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    void *output,
    uint64_t output_stride,
    uint64_t output_elem_bytes,
    const cute_tensor_t *bias,
    float *a_scale,
    float *b_scale,
    int scale_type,
    int bias_mode,
    int transpose,
    cute_llama_block_workspace_t *ws,
    cute_post_op_fn post_op,
    void *post_ctx)
{
    cute_tensor_t fallback_bias;
    const cute_tensor_t *matmul_bias = bias;
    if (matmul_bias == NULL || matmul_bias->data == NULL) {
        assert(ws->zero_bias != NULL);
        fallback_bias.data = ws->zero_bias;
        fallback_bias.stride = b->cols * 4;
        fallback_bias.rows = a->rows;
        fallback_bias.cols = b->cols;
        fallback_bias.dtype = a->dtype;
        matmul_bias = &fallback_bias;
    }

    if (cfg->use_pipeline) {
        cute_tiled_matmul_pipeline_ex(a, b, output, output_stride,
                                      output_elem_bytes, matmul_bias,
                                      a_scale, b_scale, scale_type,
                                      bias_mode, transpose,
                                      ws->scratch0, ws->scratch1,
                                      post_op, post_ctx);
    } else {
        cute_tiled_matmul_no_pipeline_ex(a, b, output, output_stride,
                                         output_elem_bytes, matmul_bias,
                                         a_scale, b_scale, scale_type,
                                         bias_mode, transpose,
                                         ws->scratch0,
                                         post_op, post_ctx);
    }
}

static inline void cute_llama_matmul_row_post(
    const cute_llama_block_config_t *cfg,
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    void *output,
    uint64_t output_stride,
    uint64_t output_elem_bytes,
    const cute_tensor_t *bias,
    cute_llama_block_workspace_t *ws,
    cute_post_op_fn post_op,
    void *post_ctx)
{
    cute_tensor_t fallback_bias;
    const cute_tensor_t *matmul_bias = bias;
    if (matmul_bias == NULL || matmul_bias->data == NULL) {
        assert(ws->zero_bias != NULL);
        fallback_bias.data = ws->zero_bias;
        fallback_bias.stride = b->cols * 4;
        fallback_bias.rows = a->rows;
        fallback_bias.cols = b->cols;
        fallback_bias.dtype = a->dtype;
        matmul_bias = &fallback_bias;
    }

    if (cfg->use_pipeline) {
        cute_tiled_matmul_row_block_pipeline_ex(
            a, b, output, output_stride, output_elem_bytes, matmul_bias,
            CUTE_TILE_M, NULL, NULL, CUTE_SCALE_NONE, CUTE_BIAS_ZERO, 0,
            ws->scratch0, ws->scratch1, post_op, post_ctx);
    } else {
        cute_tiled_matmul_row_block_no_pipeline_ex(
            a, b, output, output_stride, output_elem_bytes, matmul_bias,
            CUTE_TILE_M, NULL, NULL, CUTE_SCALE_NONE, CUTE_BIAS_ZERO, 0,
            ws->scratch0, post_op, post_ctx);
    }
}

static inline void cute_llama_project_qkv(
    const cute_llama_block_config_t *cfg,
    cute_llama_block_workspace_t *ws)
{
    int seq = cfg->seq_len;
    int embed = cfg->embed_dim;
    int q_dim = cfg->n_head_q * cfg->key_dim;
    int kv_dim = cfg->n_head_kv * cfg->key_dim;
    int v_dim = cfg->n_head_kv * cfg->value_dim;

    cute_tensor_t a = {ws->attn_norm_q8, (uint64_t)embed, seq, embed,
                       CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, seq, q_dim, CUTEDataTypeI8I8I32};
    cute_rope_ctx_t rope_ctx = {
        .pos = 0,
        .rope_theta = cfg->rope_theta,
        .key_dim = cfg->key_dim,
    };

    cute_tensor_t q_weight = {(void *)cfg->proj_q_weight, (uint64_t)embed,
                              embed, q_dim, CUTEDataTypeI8I8I32};
    cute_llama_matmul_post(cfg, &a, &q_weight,
                           ws->q_bf16, (uint64_t)q_dim * sizeof(uint16_t),
                           sizeof(uint16_t), &bias,
                           ws->attn_norm_scale,
                           (float *)cfg->proj_q_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, ws,
                           cute_post_dequant_rope_bf16cvt, &rope_ctx);

    cute_tensor_t k_weight = {(void *)cfg->proj_k_weight, (uint64_t)embed,
                              embed, kv_dim, CUTEDataTypeI8I8I32};
    bias.cols = kv_dim;
    cute_llama_matmul_post(cfg, &a, &k_weight,
                           ws->k_bf16, (uint64_t)kv_dim * sizeof(uint16_t),
                           sizeof(uint16_t), &bias,
                           ws->attn_norm_scale,
                           (float *)cfg->proj_k_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, ws,
                           cute_post_dequant_rope_bf16cvt, &rope_ctx);

    cute_tensor_t v_weight = {(void *)cfg->proj_v_weight, (uint64_t)embed,
                              embed, v_dim, CUTEDataTypeI8I8I32};
    bias.cols = v_dim;
    cute_llama_matmul_post(cfg, &a, &v_weight,
                           ws->v_bf16_t, (uint64_t)seq * sizeof(uint16_t),
                           sizeof(uint16_t), &bias,
                           ws->attn_norm_scale,
                           (float *)cfg->proj_v_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 1, ws,
                           cute_post_dequant_bf16cvt, NULL);
}

static inline void cute_llama_attention(
    const cute_llama_block_config_t *cfg,
    cute_llama_block_workspace_t *ws)
{
    int seq = cfg->seq_len;
    int embed = cfg->embed_dim;
    int q_dim = cfg->n_head_q * cfg->key_dim;
    int kv_dim = cfg->n_head_kv * cfg->key_dim;
    int q_per_kv = cfg->n_head_q / cfg->n_head_kv;

    cute_tensor_t score_bias = {NULL, 0, seq, seq, CUTEDataTypeBF16BF16F32};
    cute_softmax_ctx_t softmax_ctx = {
        .pos = 0,
        .bitmask = cfg->causal_mask,
        .max_ctx_len = cfg->max_ctx_len,
        .kv_scale = cfg->kv_scale,
    };

    for (int h = 0; h < cfg->n_head_q; h++) {
        int kv_head = h / q_per_kv;
        uint16_t *q_head = ws->q_bf16 + h * cfg->key_dim;
        uint16_t *k_head = ws->k_bf16 + kv_head * cfg->key_dim;
        uint16_t *score_head =
            ws->scores_bf16 + (size_t)h * seq * seq;

        cute_tensor_t q = {q_head, (uint64_t)q_dim * sizeof(uint16_t),
                           seq, cfg->key_dim, CUTEDataTypeBF16BF16F32};
        cute_tensor_t k = {k_head, (uint64_t)kv_dim * sizeof(uint16_t),
                           cfg->key_dim, seq, CUTEDataTypeBF16BF16F32};

        cute_llama_matmul_row_post(cfg, &q, &k,
                                   score_head,
                                   (uint64_t)seq * sizeof(uint16_t),
                                   sizeof(uint16_t),
                                   &score_bias, ws,
                                   cute_post_masked_softmax_kvscale_bf16cvt,
                                   &softmax_ctx);
    }

    for (int h = 0; h < cfg->n_head_q; h++) {
        int kv_head = h / q_per_kv;
        uint16_t *score_head =
            ws->scores_bf16 + (size_t)h * seq * seq;
        uint16_t *v_head =
            ws->v_bf16_t + (size_t)kv_head * cfg->value_dim * seq;
        float *ctx_head = ws->attn_context_f32 + h * cfg->value_dim;

        cute_tensor_t scores = {score_head, (uint64_t)seq * sizeof(uint16_t),
                                seq, seq, CUTEDataTypeBF16BF16F32};
        cute_tensor_t value = {v_head, (uint64_t)seq * sizeof(uint16_t),
                               seq, cfg->value_dim, CUTEDataTypeBF16BF16F32};
        cute_tensor_t ctx_bias = {NULL, 0, seq, cfg->value_dim,
                                  CUTEDataTypeBF16BF16F32};

        cute_llama_matmul_post(cfg, &scores, &value,
                               ctx_head, (uint64_t)embed * sizeof(float),
                               sizeof(float), &ctx_bias,
                               NULL, NULL, CUTE_SCALE_NONE,
                               CUTE_BIAS_ZERO, 0, ws,
                               NULL, NULL);
    }
}

static inline void cute_llama_ffn(
    const cute_llama_block_config_t *cfg,
    cute_llama_block_workspace_t *ws,
    const float *residual,
    float *output)
{
    int seq = cfg->seq_len;
    int embed = cfg->embed_dim;
    int ffn = cfg->ffn_dim;

    cute_tensor_t a = {ws->ffn_norm_q8, (uint64_t)embed,
                       seq, embed, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {NULL, 0, seq, ffn, CUTEDataTypeI8I8I32};

    cute_tensor_t gate_weight = {(void *)cfg->ffn_gate_weight,
                                 (uint64_t)embed,
                                 embed, ffn, CUTEDataTypeI8I8I32};
    cute_llama_matmul_post(cfg, &a, &gate_weight,
                           ws->ffn_gate_f32,
                           (uint64_t)ffn * sizeof(float),
                           sizeof(float), &bias,
                           ws->ffn_norm_scale,
                           (float *)cfg->ffn_gate_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, ws,
                           cute_post_dequant_silu, NULL);

    cute_llama_zero_f32(ws->ffn_up_scale, seq);
    cute_hadamard_ctx_t hadamard_ctx = {
        .lhs = ws->ffn_gate_f32,
        .lhs_stride = (uint64_t)ffn * sizeof(float),
        .output_absmax = ws->ffn_up_scale,
    };
    cute_tensor_t up_weight = {(void *)cfg->ffn_up_weight,
                               (uint64_t)embed,
                               embed, ffn, CUTEDataTypeI8I8I32};
    cute_llama_matmul_post(cfg, &a, &up_weight,
                           ws->ffn_up_f32,
                           (uint64_t)ffn * sizeof(float),
                           sizeof(float), &bias,
                           ws->ffn_norm_scale,
                           (float *)cfg->ffn_up_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, ws,
                           cute_post_dequant_hadamard, &hadamard_ctx);

    cute_smoothquant(ws->ffn_up_f32, seq, ffn,
                     ws->ffn_up_q8, ws->ffn_up_scale, false);

    cute_tensor_t down_a = {ws->ffn_up_q8, (uint64_t)ffn,
                            seq, ffn, CUTEDataTypeI8I8I32};
    cute_tensor_t down_weight = {(void *)cfg->ffn_down_weight,
                                 (uint64_t)ffn,
                                 ffn, embed, CUTEDataTypeI8I8I32};
    cute_tensor_t down_bias = {NULL, 0, seq, embed, CUTEDataTypeI8I8I32};
    cute_resadd_ctx_t resadd_ctx = {
        .residual = residual,
        .residual_stride = (uint64_t)embed * sizeof(float),
    };
    cute_llama_matmul_post(cfg, &down_a, &down_weight,
                           output, (uint64_t)embed * sizeof(float),
                           sizeof(float), &down_bias,
                           ws->ffn_up_scale,
                           (float *)cfg->ffn_down_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, ws,
                           cute_post_dequant_resadd, &resadd_ctx);
}

static inline void cute_llama_block(const cute_llama_block_config_t *cfg,
                                    cute_llama_block_workspace_t *ws,
                                    const float *input,
                                    float *output)
{
    int seq = cfg->seq_len;
    int embed = cfg->embed_dim;

    cute_rmsnorm_with_scale(input, ws->attn_norm_f32,
                            cfg->attn_norm_weight,
                            ws->attn_norm_scale,
                            cfg->rms_epsilon, 1, seq, embed);
    cute_smoothquant(ws->attn_norm_f32, seq, embed,
                     ws->attn_norm_q8, ws->attn_norm_scale, false);

    cute_llama_project_qkv(cfg, ws);
    cute_llama_attention(cfg, ws);

    cute_smoothquant(ws->attn_context_f32, seq, embed,
                     ws->attn_q8, ws->attn_scale, true);

    cute_tensor_t attn_a = {ws->attn_q8, (uint64_t)embed,
                            seq, embed, CUTEDataTypeI8I8I32};
    cute_tensor_t proj_o_weight = {(void *)cfg->proj_o_weight,
                                   (uint64_t)embed,
                                   embed, embed, CUTEDataTypeI8I8I32};
    cute_tensor_t proj_o_bias = {NULL, 0, seq, embed, CUTEDataTypeI8I8I32};
    cute_resadd_ctx_t proj_o_resadd = {
        .residual = input,
        .residual_stride = (uint64_t)embed * sizeof(float),
    };
    cute_llama_matmul_post(cfg, &attn_a, &proj_o_weight,
                           ws->proj_o_f32,
                           (uint64_t)embed * sizeof(float),
                           sizeof(float), &proj_o_bias,
                           ws->attn_scale,
                           (float *)cfg->proj_o_weight_scale,
                           CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                           CUTE_BIAS_ZERO, 0, ws,
                           cute_post_dequant_resadd, &proj_o_resadd);

    cute_rmsnorm_with_scale(ws->proj_o_f32, ws->ffn_norm_f32,
                            cfg->ffn_norm_weight,
                            ws->ffn_norm_scale,
                            cfg->rms_epsilon, 1, seq, embed);
    cute_smoothquant(ws->ffn_norm_f32, seq, embed,
                     ws->ffn_norm_q8, ws->ffn_norm_scale, false);

    cute_llama_ffn(cfg, ws, ws->proj_o_f32, output);
}

#endif /* CUTE_LLAMA_H */
