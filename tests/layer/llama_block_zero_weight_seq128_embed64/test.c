#include <stdint.h>

#include "cute_llama.h"
#include "golden/manual/layer/llama_block_zero_weight_seq128_embed64/golden_llama_block_inputs.h"
#include "tests/primitive/primitive_test_utils.h"

#define TEST_SEQ_LEN 128
#define TEST_EMBED_DIM 64
#define TEST_KEY_DIM 64
#define TEST_VALUE_DIM 64
#define TEST_N_HEAD_Q 1
#define TEST_N_HEAD_KV 1
#define TEST_FFN_DIM 64
#define TEST_MAX_CTX_LEN 128

static float output[TEST_SEQ_LEN][TEST_EMBED_DIM] __attribute__((used, aligned(64)));

static int8_t proj_q_weight[TEST_N_HEAD_Q * TEST_KEY_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t proj_k_weight[TEST_N_HEAD_KV * TEST_KEY_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t proj_o_weight[TEST_EMBED_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_gate_weight[TEST_FFN_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_up_weight[TEST_FFN_DIM][TEST_EMBED_DIM] CUTE_TEST_ALIGN;
static int8_t ffn_down_weight[TEST_EMBED_DIM][TEST_FFN_DIM] CUTE_TEST_ALIGN;

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
static uint32_t zero_bias[TEST_SEQ_LEN][TEST_FFN_DIM] CUTE_TEST_ALIGN;

int main(void)
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
        .proj_q_weight = &proj_q_weight[0][0],
        .proj_q_weight_scale = golden_llama_proj_q_scale,
        .proj_k_weight = &proj_k_weight[0][0],
        .proj_k_weight_scale = golden_llama_proj_k_scale,
        .proj_v_weight = golden_llama_proj_v_weight,
        .proj_v_weight_scale = golden_llama_proj_v_scale,
        .proj_o_weight = &proj_o_weight[0][0],
        .proj_o_weight_scale = golden_llama_proj_o_scale,
        .ffn_norm_weight = golden_llama_ffn_norm_weight,
        .ffn_gate_weight = &ffn_gate_weight[0][0],
        .ffn_gate_weight_scale = golden_llama_ffn_gate_scale,
        .ffn_up_weight = &ffn_up_weight[0][0],
        .ffn_up_weight_scale = golden_llama_ffn_up_scale,
        .ffn_down_weight = &ffn_down_weight[0][0],
        .ffn_down_weight_scale = golden_llama_ffn_down_scale,
        .rope_theta = golden_llama_rope_theta,
        .causal_mask = golden_llama_causal_mask,
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
        .zero_bias = zero_bias,
    };

    cute_llama_block(&cfg, &ws, golden_llama_input, &output[0][0]);
    cute_test_host_verify_barrier();
    return 0;
}
