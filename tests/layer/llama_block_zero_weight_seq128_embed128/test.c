#include <stdint.h>

#include "cute_llama.h"
#include "tests/primitive/primitive_test_utils.h"

#define TEST_SEQ_LEN 128
#define TEST_EMBED_DIM 128
#define TEST_KEY_DIM 64
#define TEST_VALUE_DIM 64
#define TEST_N_HEAD_Q 2
#define TEST_N_HEAD_KV 1
#define TEST_FFN_DIM 128
#define TEST_MAX_CTX_LEN 128

static float input[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static float output[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;

static float attn_norm_weight[TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static float ffn_norm_weight[TEST_EMBED_DIM] CUTE_TEST_ALIGN;

static int8_t proj_q_weight[TEST_N_HEAD_Q * TEST_KEY_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t proj_k_weight[TEST_N_HEAD_KV * TEST_KEY_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t proj_v_weight[TEST_N_HEAD_KV * TEST_VALUE_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t proj_o_weight[TEST_EMBED_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_gate_weight[TEST_FFN_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_up_weight[TEST_FFN_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_down_weight[TEST_EMBED_DIM][TEST_FFN_DIM] CUTE_TEST_ALIGN;

static float proj_q_scale[1] CUTE_TEST_ALIGN = {0.001f};
static float proj_k_scale[1] CUTE_TEST_ALIGN = {0.001f};
static float proj_v_scale[1] CUTE_TEST_ALIGN = {0.001f};
static float proj_o_scale[1] CUTE_TEST_ALIGN = {0.001f};
static float ffn_gate_scale[1] CUTE_TEST_ALIGN = {0.001f};
static float ffn_up_scale[1] CUTE_TEST_ALIGN = {0.001f};
static float ffn_down_scale[1] CUTE_TEST_ALIGN = {0.001f};

static float rope_theta[TEST_KEY_DIM / 2] CUTE_TEST_ALIGN;
static int8_t causal_mask[TEST_SEQ_LEN * ((TEST_MAX_CTX_LEN + 7) / 8)] CUTE_TEST_ALIGN;

static float attn_norm_f32[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t attn_norm_q8[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static float attn_norm_scale[TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static uint16_t q_bf16[TEST_SEQ_LEN][TEST_N_HEAD_Q * TEST_KEY_DIM] CUTE_TEST_ALIGN;
static uint16_t k_bf16[TEST_SEQ_LEN][TEST_N_HEAD_KV * TEST_KEY_DIM] CUTE_TEST_ALIGN;
static uint16_t v_bf16_t[TEST_N_HEAD_KV * TEST_VALUE_DIM][TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static uint16_t scores_bf16[TEST_N_HEAD_Q][TEST_SEQ_LEN][TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static float attn_context_f32[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t attn_q8[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static float attn_scale[TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static float proj_o_f32[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static float ffn_norm_f32[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_norm_q8[TEST_SEQ_LEN][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static float ffn_norm_scale[TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static float ffn_gate_f32[TEST_SEQ_LEN][TEST_FFN_DIM] CUTE_TEST_ALIGN;
static float ffn_up_f32[TEST_SEQ_LEN][TEST_FFN_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_up_q8[TEST_SEQ_LEN][TEST_FFN_DIM] CUTE_TEST_ALIGN;
static float ffn_up_token_scale[TEST_SEQ_LEN] CUTE_TEST_ALIGN;
static float scratch[2][CUTE_TILE_M][TEST_SEQ_LEN] CUTE_TEST_ALIGN;

static inline float test_input_value(int row, int col)
{
    int v = ((row * 17 + col * 5) % 97) + 1;
    return (float)v * 0.0009765625f;
}

static inline void init_causal_mask(void)
{
    int stride = (TEST_MAX_CTX_LEN + 7) / 8;
    for (int i = 0; i < TEST_SEQ_LEN * stride; i++) {
        causal_mask[i] = 0;
    }
    for (int row = 0; row < TEST_SEQ_LEN; row++) {
        for (int col = 0; col <= row; col++) {
            int bit = row * stride * 8 + col;
            causal_mask[bit / 8] |= (int8_t)(1u << (bit % 8));
        }
    }
}

static inline void init_inputs(void)
{
    for (int row = 0; row < TEST_SEQ_LEN; row++) {
        for (int col = 0; col < TEST_EMBED_DIM; col++) {
            input[row][col] = test_input_value(row, col);
        }
    }
    for (int col = 0; col < TEST_EMBED_DIM; col++) {
        attn_norm_weight[col] = 1.0f;
        ffn_norm_weight[col] = 1.0f;
    }
    for (int k = 0; k < TEST_KEY_DIM / 2; k++) {
        rope_theta[k] = 1.0f;
    }
    init_causal_mask();
}

int main(void)
{
    init_inputs();

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
        .attn_norm_weight = attn_norm_weight,
        .proj_q_weight = &proj_q_weight[0][0],
        .proj_q_weight_scale = proj_q_scale,
        .proj_k_weight = &proj_k_weight[0][0],
        .proj_k_weight_scale = proj_k_scale,
        .proj_v_weight = &proj_v_weight[0][0],
        .proj_v_weight_scale = proj_v_scale,
        .proj_o_weight = &proj_o_weight[0][0],
        .proj_o_weight_scale = proj_o_scale,
        .ffn_norm_weight = ffn_norm_weight,
        .ffn_gate_weight = &ffn_gate_weight[0][0],
        .ffn_gate_weight_scale = ffn_gate_scale,
        .ffn_up_weight = &ffn_up_weight[0][0],
        .ffn_up_weight_scale = ffn_up_scale,
        .ffn_down_weight = &ffn_down_weight[0][0],
        .ffn_down_weight_scale = ffn_down_scale,
        .rope_theta = rope_theta,
        .causal_mask = causal_mask,
    };

    cute_llama_block_workspace_t ws = {
        .attn_norm_f32 = &attn_norm_f32[0][0],
        .attn_norm_q8 = &attn_norm_q8[0][0],
        .attn_norm_scale = attn_norm_scale,
        .q_bf16 = &q_bf16[0][0],
        .k_bf16 = &k_bf16[0][0],
        .v_bf16_t = &v_bf16_t[0][0],
        .scores_bf16 = &scores_bf16[0][0][0],
        .attn_context_f32 = &attn_context_f32[0][0],
        .attn_q8 = &attn_q8[0][0],
        .attn_scale = attn_scale,
        .proj_o_f32 = &proj_o_f32[0][0],
        .ffn_norm_f32 = &ffn_norm_f32[0][0],
        .ffn_norm_q8 = &ffn_norm_q8[0][0],
        .ffn_norm_scale = ffn_norm_scale,
        .ffn_gate_f32 = &ffn_gate_f32[0][0],
        .ffn_up_f32 = &ffn_up_f32[0][0],
        .ffn_up_q8 = &ffn_up_q8[0][0],
        .ffn_up_scale = ffn_up_token_scale,
        .scratch0 = scratch[0],
        .scratch1 = scratch[1],
    };

    cute_llama_block(&cfg, &ws, &input[0][0], &output[0][0]);
    cute_test_host_verify_barrier();
    return 0;
}
