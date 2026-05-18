#ifndef CUTE_FUSION_MATMUL_POST_UTILS_H
#define CUTE_FUSION_MATMUL_POST_UTILS_H

#include <math.h>
#include <stdint.h>

#include "cute_fusion.h"
#include "tests/primitive/primitive_test_utils.h"
#include "../runtime/runtime_matmul_i8_128_128_128_zeroinit/matmul_value_mnk_128_128_128_zeroinit.h"

#define CUTE_FUSION_OUTPUT_M APPLICATION_M
#define CUTE_FUSION_OUTPUT_N APPLICATION_N
#define CUTE_FUSION_N64 64
#define CUTE_FUSION_N128 128
#define CUTE_FUSION_SOFTMAX_K64 64
#define CUTE_FUSION_CONTEXT_N64 64
#define CUTE_FUSION_CONTEXT_K128 128

static float input_scale[APPLICATION_M] CUTE_TEST_ALIGN;
static float weight_scale[1] CUTE_TEST_ALIGN = {0.001f};

static _Float16 softmax_a[APPLICATION_M][CUTE_FUSION_SOFTMAX_K64] CUTE_TEST_ALIGN;
static _Float16 softmax_b[CUTE_FUSION_N64][CUTE_FUSION_SOFTMAX_K64] CUTE_TEST_ALIGN;
static float softmax_zero_bias[APPLICATION_M][CUTE_FUSION_N64] CUTE_TEST_ALIGN;
static _Float16 softmax128_b[CUTE_FUSION_N128][CUTE_FUSION_SOFTMAX_K64] CUTE_TEST_ALIGN;
static float softmax128_zero_bias[APPLICATION_M][CUTE_FUSION_N128] CUTE_TEST_ALIGN;
static _Float16 attention_scores[APPLICATION_M][CUTE_FUSION_CONTEXT_K128] CUTE_TEST_ALIGN;
static _Float16 attention_value[CUTE_FUSION_CONTEXT_N64][CUTE_FUSION_CONTEXT_K128] CUTE_TEST_ALIGN;
static float attention_zero_bias[APPLICATION_M][CUTE_FUSION_CONTEXT_N64] CUTE_TEST_ALIGN;

static inline float cute_fusion_input_scale_value(int row)
{
    return 0.001f + (float)(row % 17) * 0.00001f;
}

static inline float cute_fusion_residual_value(int row, int col)
{
    int v = ((row * 13 + col * 7) % 257) - 128;
    return (float)v * 0.00025f;
}

static inline float cute_fusion_hadamard_lhs_value(int row, int col)
{
    int v = ((row * 11 + col * 5) % 127) - 63;
    return (float)v * 0.03125f;
}

static inline float cute_fusion_rope_theta_value(int k)
{
    const float head_dim = (float)CUTE_FUSION_N64;
    return powf(10000.0f, -2.0f * (float)k / head_dim);
}

static inline _Float16 cute_fusion_softmax_a_value(int row, int col)
{
    int v = ((row * 5 + col * 3) % 17) - 8;
    return (_Float16)((float)v * 0.0625f);
}

static inline _Float16 cute_fusion_softmax_b_value(int row, int col)
{
    int v = ((row * 7 + col * 11) % 19) - 9;
    return (_Float16)((float)v * 0.0625f);
}

static inline _Float16 cute_fusion_attention_score_value(int row, int col)
{
    if (col > row) {
        return (_Float16)0.0f;
    }
    int v = ((row * 3 + col * 5) % 17) + 1;
    return (_Float16)((float)v * 0.00390625f);
}

static inline _Float16 cute_fusion_attention_value_value(int row, int col)
{
    int v = ((row * 13 + col * 7) % 31) - 15;
    return (_Float16)((float)v * 0.03125f);
}

static inline void cute_fusion_init_scales(void)
{
    (void)c;
    (void)gloden_c;

    for (int row = 0; row < APPLICATION_M; row++) {
        input_scale[row] = cute_fusion_input_scale_value(row);
    }
}

static inline void cute_fusion_init_residual(float residual[APPLICATION_M][APPLICATION_N])
{
    for (int row = 0; row < APPLICATION_M; row++) {
        for (int col = 0; col < APPLICATION_N; col++) {
            residual[row][col] = cute_fusion_residual_value(row, col);
        }
    }
}

static inline void cute_fusion_init_hadamard_lhs(
    float lhs[APPLICATION_M][APPLICATION_N])
{
    for (int row = 0; row < APPLICATION_M; row++) {
        for (int col = 0; col < APPLICATION_N; col++) {
            lhs[row][col] = cute_fusion_hadamard_lhs_value(row, col);
        }
    }
}

static inline void cute_fusion_init_rope_theta(float theta[CUTE_FUSION_N64 / 2])
{
    for (int k = 0; k < CUTE_FUSION_N64 / 2; k++) {
        theta[k] = cute_fusion_rope_theta_value(k);
    }
}

static inline void cute_fusion_zero_f32(float *data, int count)
{
    for (int i = 0; i < count; i++) {
        data[i] = 0.0f;
    }
}

static inline void cute_fusion_init_causal_mask(int8_t *mask, int rows, int cols)
{
    int stride = (cols + 7) / 8;
    for (int i = 0; i < rows * stride; i++) {
        mask[i] = 0;
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col <= row && col < cols; col++) {
            int bit = row * stride * 8 + col;
            mask[bit / 8] |= (int8_t)(1u << (bit % 8));
        }
    }
}

static inline void cute_fusion_init_softmax_inputs(void)
{
    (void)c;
    (void)gloden_c;

    for (int row = 0; row < APPLICATION_M; row++) {
        for (int col = 0; col < CUTE_FUSION_SOFTMAX_K64; col++) {
            softmax_a[row][col] = cute_fusion_softmax_a_value(row, col);
        }
    }
    for (int row = 0; row < CUTE_FUSION_N64; row++) {
        for (int col = 0; col < CUTE_FUSION_SOFTMAX_K64; col++) {
            softmax_b[row][col] = cute_fusion_softmax_b_value(row, col);
        }
    }
    cute_fusion_zero_f32(&softmax_zero_bias[0][0],
                         APPLICATION_M * CUTE_FUSION_N64);
}

static inline void cute_fusion_init_softmax128_inputs(void)
{
    cute_fusion_init_softmax_inputs();

    for (int row = 0; row < CUTE_FUSION_N128; row++) {
        for (int col = 0; col < CUTE_FUSION_SOFTMAX_K64; col++) {
            softmax128_b[row][col] = cute_fusion_softmax_b_value(row, col);
        }
    }
    cute_fusion_zero_f32(&softmax128_zero_bias[0][0],
                         APPLICATION_M * CUTE_FUSION_N128);
}

static inline void cute_fusion_init_attention_context_inputs(void)
{
    (void)c;
    (void)gloden_c;

    for (int row = 0; row < APPLICATION_M; row++) {
        for (int col = 0; col < CUTE_FUSION_CONTEXT_K128; col++) {
            attention_scores[row][col] =
                cute_fusion_attention_score_value(row, col);
        }
    }
    for (int row = 0; row < CUTE_FUSION_CONTEXT_N64; row++) {
        for (int col = 0; col < CUTE_FUSION_CONTEXT_K128; col++) {
            attention_value[row][col] =
                cute_fusion_attention_value_value(row, col);
        }
    }
    cute_fusion_zero_f32(&attention_zero_bias[0][0],
                         APPLICATION_M * CUTE_FUSION_CONTEXT_N64);
}

static inline void cute_fusion_matmul_notile(int32_t acc[APPLICATION_M][APPLICATION_N])
{
    uint64_t tid = cute_matmul(
        a, APPLICATION_K * sizeof(a[0][0]),
        b, APPLICATION_K * sizeof(b[0][0]),
        d, APPLICATION_N * sizeof(d[0][0]),
        acc, APPLICATION_N * sizeof(acc[0][0]),
        APPLICATION_M, APPLICATION_N, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, TRANSPOSE_RESULT, 0);
    cute_wait_task(tid);
}

static inline void cute_fusion_matmul_transpose_notile(
    int32_t acc[APPLICATION_N][APPLICATION_M])
{
    uint64_t tid = cute_matmul(
        a, APPLICATION_K * sizeof(a[0][0]),
        b, APPLICATION_K * sizeof(b[0][0]),
        d, APPLICATION_N * sizeof(d[0][0]),
        acc, APPLICATION_M * sizeof(acc[0][0]),
        APPLICATION_M, APPLICATION_N, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, 1, 0);
    cute_wait_task(tid);
}

static inline void cute_fusion_matmul_n64_notile(
    int32_t acc[APPLICATION_M][CUTE_FUSION_N64])
{
    uint64_t tid = cute_matmul(
        a, APPLICATION_K * sizeof(a[0][0]),
        b, APPLICATION_K * sizeof(b[0][0]),
        d, APPLICATION_N * sizeof(d[0][0]),
        acc, CUTE_FUSION_N64 * sizeof(acc[0][0]),
        APPLICATION_M, CUTE_FUSION_N64, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, TRANSPOSE_RESULT, 0);
    cute_wait_task(tid);
}

static inline void cute_fusion_softmax_matmul_notile(
    float scores[APPLICATION_M][CUTE_FUSION_N64])
{
    uint64_t tid = cute_matmul(
        softmax_a, CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_a[0][0]),
        softmax_b, CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_b[0][0]),
        softmax_zero_bias, CUTE_FUSION_N64 * sizeof(softmax_zero_bias[0][0]),
        scores, CUTE_FUSION_N64 * sizeof(scores[0][0]),
        APPLICATION_M, CUTE_FUSION_N64, CUTE_FUSION_SOFTMAX_K64,
        CUTEDataTypeF16F16F32, CUTE_BIAS_ZERO, 0, 0);
    cute_wait_task(tid);
}

static inline void cute_fusion_softmax128_matmul_notile(
    float scores[APPLICATION_M][CUTE_FUSION_N128])
{
    uint64_t tid = cute_matmul(
        softmax_a, CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_a[0][0]),
        softmax128_b, CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax128_b[0][0]),
        softmax128_zero_bias,
        CUTE_FUSION_N128 * sizeof(softmax128_zero_bias[0][0]),
        scores, CUTE_FUSION_N128 * sizeof(scores[0][0]),
        APPLICATION_M, CUTE_FUSION_N128, CUTE_FUSION_SOFTMAX_K64,
        CUTEDataTypeF16F16F32, CUTE_BIAS_ZERO, 0, 0);
    cute_wait_task(tid);
}

static inline void cute_fusion_attention_context_matmul_notile(
    float output[APPLICATION_M][CUTE_FUSION_CONTEXT_N64])
{
    uint64_t tid = cute_matmul(
        attention_scores,
        CUTE_FUSION_CONTEXT_K128 * sizeof(attention_scores[0][0]),
        attention_value,
        CUTE_FUSION_CONTEXT_K128 * sizeof(attention_value[0][0]),
        attention_zero_bias,
        CUTE_FUSION_CONTEXT_N64 * sizeof(attention_zero_bias[0][0]),
        output,
        CUTE_FUSION_CONTEXT_N64 * sizeof(output[0][0]),
        APPLICATION_M, CUTE_FUSION_CONTEXT_N64, CUTE_FUSION_CONTEXT_K128,
        CUTEDataTypeF16F16F32, CUTE_BIAS_ZERO, 0, 0);
    cute_wait_task(tid);
}

static inline void cute_fusion_run_notile(void *output,
                                          uint64_t output_stride,
                                          cute_post_op_fn post_op,
                                          void *post_ctx)
{
    static int32_t acc[APPLICATION_M][APPLICATION_N] CUTE_TEST_ALIGN;

    cute_fusion_matmul_notile(acc);

    cute_post_call_t call = {
        .tile = {
            .src = acc,
            .dst = output,
            .src_stride = APPLICATION_N * sizeof(acc[0][0]),
            .dst_stride = output_stride,
            .rows = APPLICATION_M,
            .cols = APPLICATION_N,
            .tile_i = 0,
            .tile_j = 0,
            .row0 = 0,
            .col0 = 0,
        },
        .env = {
            .a_scale = input_scale,
            .b_scale = weight_scale,
            .scale_type = CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
            .bias_mode = BIAS_TYPE,
            .transpose = TRANSPOSE_RESULT,
        },
        .user_ctx = post_ctx,
    };
    post_op(&call);
}

static inline void cute_fusion_run_bf16cvt_transpose_notile(
    void *output,
    uint64_t output_stride)
{
    static int32_t acc[APPLICATION_N][APPLICATION_M] CUTE_TEST_ALIGN;

    cute_fusion_matmul_transpose_notile(acc);

    cute_post_call_t call = {
        .tile = {
            .src = acc,
            .dst = output,
            .src_stride = APPLICATION_M * sizeof(acc[0][0]),
            .dst_stride = output_stride,
            .rows = APPLICATION_N,
            .cols = APPLICATION_M,
            .tile_i = 0,
            .tile_j = 0,
            .row0 = 0,
            .col0 = 0,
        },
        .env = {
            .a_scale = input_scale,
            .b_scale = weight_scale,
            .scale_type = CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
            .bias_mode = BIAS_TYPE,
            .transpose = 1,
        },
        .user_ctx = NULL,
    };
    cute_post_dequant_bf16cvt(&call);
}

static inline void cute_fusion_run_n64_notile(void *output,
                                              uint64_t output_stride,
                                              cute_post_op_fn post_op,
                                              void *post_ctx)
{
    static int32_t acc[APPLICATION_M][CUTE_FUSION_N64] CUTE_TEST_ALIGN;

    cute_fusion_matmul_n64_notile(acc);

    cute_post_call_t call = {
        .tile = {
            .src = acc,
            .dst = output,
            .src_stride = CUTE_FUSION_N64 * sizeof(acc[0][0]),
            .dst_stride = output_stride,
            .rows = APPLICATION_M,
            .cols = CUTE_FUSION_N64,
            .tile_i = 0,
            .tile_j = 0,
            .row0 = 0,
            .col0 = 0,
        },
        .env = {
            .a_scale = input_scale,
            .b_scale = weight_scale,
            .scale_type = CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
            .bias_mode = BIAS_TYPE,
            .transpose = TRANSPOSE_RESULT,
        },
        .user_ctx = post_ctx,
    };
    post_op(&call);
}

static inline void cute_fusion_run_softmax_notile(void *output,
                                                  uint64_t output_stride,
                                                  cute_post_op_fn post_op,
                                                  void *post_ctx)
{
    static float scores[APPLICATION_M][CUTE_FUSION_N64] CUTE_TEST_ALIGN;

    cute_fusion_softmax_matmul_notile(scores);

    cute_post_call_t call = {
        .tile = {
            .src = scores,
            .dst = output,
            .src_stride = CUTE_FUSION_N64 * sizeof(scores[0][0]),
            .dst_stride = output_stride,
            .rows = APPLICATION_M,
            .cols = CUTE_FUSION_N64,
            .tile_i = 0,
            .tile_j = 0,
            .row0 = 0,
            .col0 = 0,
        },
        .env = {
            .a_scale = NULL,
            .b_scale = NULL,
            .scale_type = CUTE_SCALE_NONE,
            .bias_mode = CUTE_BIAS_ZERO,
            .transpose = 0,
        },
        .user_ctx = post_ctx,
    };
    post_op(&call);
}

static inline void cute_fusion_run_softmax128_notile(void *output,
                                                     uint64_t output_stride,
                                                     cute_post_op_fn post_op,
                                                     void *post_ctx)
{
    static float scores[APPLICATION_M][CUTE_FUSION_N128] CUTE_TEST_ALIGN;

    cute_fusion_softmax128_matmul_notile(scores);

    cute_post_call_t call = {
        .tile = {
            .src = scores,
            .dst = output,
            .src_stride = CUTE_FUSION_N128 * sizeof(scores[0][0]),
            .dst_stride = output_stride,
            .rows = APPLICATION_M,
            .cols = CUTE_FUSION_N128,
            .tile_i = 0,
            .tile_j = 0,
            .row0 = 0,
            .col0 = 0,
        },
        .env = {
            .a_scale = NULL,
            .b_scale = NULL,
            .scale_type = CUTE_SCALE_NONE,
            .bias_mode = CUTE_BIAS_ZERO,
            .transpose = 0,
        },
        .user_ctx = post_ctx,
    };
    post_op(&call);
}

static inline void cute_fusion_run_nopipeline(void *output,
                                              uint64_t output_stride,
                                              uint64_t output_elem_bytes,
                                              cute_post_op_fn post_op,
                                              void *post_ctx)
{
    static int32_t scratch[CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {a, APPLICATION_K * sizeof(a[0][0]),
                        APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K * sizeof(b[0][0]),
                        APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {d, APPLICATION_N * sizeof(d[0][0]),
                          APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_no_pipeline_ex(&ta, &tb,
                                     output, output_stride, output_elem_bytes,
                                     &bias,
                                     input_scale, weight_scale,
                                     CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                                     BIAS_TYPE, TRANSPOSE_RESULT,
                                     scratch,
                                     post_op, post_ctx);
}

static inline void cute_fusion_run_bf16cvt_transpose_nopipeline(
    void *output,
    uint64_t output_stride,
    uint64_t output_elem_bytes)
{
    static int32_t scratch[CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {a, APPLICATION_K * sizeof(a[0][0]),
                        APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K * sizeof(b[0][0]),
                        APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {d, APPLICATION_N * sizeof(d[0][0]),
                          APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_no_pipeline_ex(&ta, &tb,
                                     output, output_stride, output_elem_bytes,
                                     &bias,
                                     input_scale, weight_scale,
                                     CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                                     BIAS_TYPE, 1,
                                     scratch,
                                     cute_post_dequant_bf16cvt, NULL);
}

static inline void cute_fusion_run_n64_nopipeline(void *output,
                                                  uint64_t output_stride,
                                                  uint64_t output_elem_bytes,
                                                  cute_post_op_fn post_op,
                                                  void *post_ctx)
{
    static int32_t scratch[CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {a, APPLICATION_K * sizeof(a[0][0]),
                        APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K * sizeof(b[0][0]),
                        APPLICATION_K, CUTE_FUSION_N64, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {d, APPLICATION_N * sizeof(d[0][0]),
                          APPLICATION_M, CUTE_FUSION_N64, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_no_pipeline_ex(&ta, &tb,
                                     output, output_stride, output_elem_bytes,
                                     &bias,
                                     input_scale, weight_scale,
                                     CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                                     BIAS_TYPE, TRANSPOSE_RESULT,
                                     scratch,
                                     post_op, post_ctx);
}

static inline void cute_fusion_run_softmax_nopipeline(void *output,
                                                      uint64_t output_stride,
                                                      uint64_t output_elem_bytes,
                                                      cute_post_op_fn post_op,
                                                      void *post_ctx)
{
    static float scratch[CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {softmax_a,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_a[0][0]),
                        APPLICATION_M, CUTE_FUSION_SOFTMAX_K64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t tb = {softmax_b,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_b[0][0]),
                        CUTE_FUSION_SOFTMAX_K64, CUTE_FUSION_N64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t bias = {softmax_zero_bias,
                          CUTE_FUSION_N64 * sizeof(softmax_zero_bias[0][0]),
                          APPLICATION_M, CUTE_FUSION_N64,
                          CUTEDataTypeF16F16F32};

    cute_tiled_matmul_no_pipeline_ex(&ta, &tb,
                                     output, output_stride, output_elem_bytes,
                                     &bias,
                                     NULL, NULL,
                                     CUTE_SCALE_NONE,
                                     CUTE_BIAS_ZERO, 0,
                                     scratch,
                                     post_op, post_ctx);
}

static inline void cute_fusion_run_softmax128_nopipeline(void *output,
                                                         uint64_t output_stride,
                                                         uint64_t output_elem_bytes,
                                                         cute_post_op_fn post_op,
                                                         void *post_ctx)
{
    static float scratch[CUTE_TILE_M][CUTE_FUSION_N128] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {softmax_a,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_a[0][0]),
                        APPLICATION_M, CUTE_FUSION_SOFTMAX_K64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t tb = {softmax128_b,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax128_b[0][0]),
                        CUTE_FUSION_SOFTMAX_K64, CUTE_FUSION_N128,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t bias = {softmax128_zero_bias,
                          CUTE_FUSION_N128 * sizeof(softmax128_zero_bias[0][0]),
                          APPLICATION_M, CUTE_FUSION_N128,
                          CUTEDataTypeF16F16F32};

    cute_tiled_matmul_row_block_no_pipeline_ex(&ta, &tb,
                                               output,
                                               output_stride,
                                               output_elem_bytes,
                                               &bias,
                                               CUTE_TILE_M,
                                               NULL, NULL,
                                               CUTE_SCALE_NONE,
                                               CUTE_BIAS_ZERO, 0,
                                               scratch,
                                               post_op, post_ctx);
}

static inline void cute_fusion_run_attention_context_nopipeline(
    float output[APPLICATION_M][CUTE_FUSION_CONTEXT_N64])
{
    static float scratch[CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {attention_scores,
                        CUTE_FUSION_CONTEXT_K128 * sizeof(attention_scores[0][0]),
                        APPLICATION_M, CUTE_FUSION_CONTEXT_K128,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t tb = {attention_value,
                        CUTE_FUSION_CONTEXT_K128 * sizeof(attention_value[0][0]),
                        CUTE_FUSION_CONTEXT_K128, CUTE_FUSION_CONTEXT_N64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t bias = {attention_zero_bias,
                          CUTE_FUSION_CONTEXT_N64 * sizeof(attention_zero_bias[0][0]),
                          APPLICATION_M, CUTE_FUSION_CONTEXT_N64,
                          CUTEDataTypeF16F16F32};

    cute_tiled_matmul_no_pipeline_ex(&ta, &tb,
                                     output,
                                     CUTE_FUSION_CONTEXT_N64 *
                                         sizeof(output[0][0]),
                                     sizeof(output[0][0]),
                                     &bias,
                                     NULL, NULL,
                                     CUTE_SCALE_NONE,
                                     CUTE_BIAS_ZERO, 0,
                                     scratch,
                                     NULL, NULL);
}

static inline void cute_fusion_run_pipeline(void *output,
                                            uint64_t output_stride,
                                            uint64_t output_elem_bytes,
                                            cute_post_op_fn post_op,
                                            void *post_ctx)
{
    static int32_t scratch[2][CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {a, APPLICATION_K * sizeof(a[0][0]),
                        APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K * sizeof(b[0][0]),
                        APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {d, APPLICATION_N * sizeof(d[0][0]),
                          APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_pipeline_ex(&ta, &tb,
                                  output, output_stride, output_elem_bytes,
                                  &bias,
                                  input_scale, weight_scale,
                                  CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                                  BIAS_TYPE, TRANSPOSE_RESULT,
                                  scratch[0], scratch[1],
                                  post_op, post_ctx);
}

static inline void cute_fusion_run_bf16cvt_transpose_pipeline(
    void *output,
    uint64_t output_stride,
    uint64_t output_elem_bytes)
{
    static int32_t scratch[2][CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {a, APPLICATION_K * sizeof(a[0][0]),
                        APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K * sizeof(b[0][0]),
                        APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {d, APPLICATION_N * sizeof(d[0][0]),
                          APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_pipeline_ex(&ta, &tb,
                                  output, output_stride, output_elem_bytes,
                                  &bias,
                                  input_scale, weight_scale,
                                  CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                                  BIAS_TYPE, 1,
                                  scratch[0], scratch[1],
                                  cute_post_dequant_bf16cvt, NULL);
}

static inline void cute_fusion_run_n64_pipeline(void *output,
                                                uint64_t output_stride,
                                                uint64_t output_elem_bytes,
                                                cute_post_op_fn post_op,
                                                void *post_ctx)
{
    static int32_t scratch[2][CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {a, APPLICATION_K * sizeof(a[0][0]),
                        APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K * sizeof(b[0][0]),
                        APPLICATION_K, CUTE_FUSION_N64, CUTEDataTypeI8I8I32};
    cute_tensor_t bias = {d, APPLICATION_N * sizeof(d[0][0]),
                          APPLICATION_M, CUTE_FUSION_N64, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_pipeline_ex(&ta, &tb,
                                  output, output_stride, output_elem_bytes,
                                  &bias,
                                  input_scale, weight_scale,
                                  CUTE_SCALE_PERTOKEN_A_PERTENSOR_B,
                                  BIAS_TYPE, TRANSPOSE_RESULT,
                                  scratch[0], scratch[1],
                                  post_op, post_ctx);
}

static inline void cute_fusion_run_softmax_pipeline(void *output,
                                                    uint64_t output_stride,
                                                    uint64_t output_elem_bytes,
                                                    cute_post_op_fn post_op,
                                                    void *post_ctx)
{
    static float scratch[2][CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {softmax_a,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_a[0][0]),
                        APPLICATION_M, CUTE_FUSION_SOFTMAX_K64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t tb = {softmax_b,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_b[0][0]),
                        CUTE_FUSION_SOFTMAX_K64, CUTE_FUSION_N64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t bias = {softmax_zero_bias,
                          CUTE_FUSION_N64 * sizeof(softmax_zero_bias[0][0]),
                          APPLICATION_M, CUTE_FUSION_N64,
                          CUTEDataTypeF16F16F32};

    cute_tiled_matmul_pipeline_ex(&ta, &tb,
                                  output, output_stride, output_elem_bytes,
                                  &bias,
                                  NULL, NULL,
                                  CUTE_SCALE_NONE,
                                  CUTE_BIAS_ZERO, 0,
                                  scratch[0], scratch[1],
                                  post_op, post_ctx);
}

static inline void cute_fusion_run_softmax128_pipeline(void *output,
                                                       uint64_t output_stride,
                                                       uint64_t output_elem_bytes,
                                                       cute_post_op_fn post_op,
                                                       void *post_ctx)
{
    static float scratch[2][CUTE_TILE_M][CUTE_FUSION_N128] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {softmax_a,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax_a[0][0]),
                        APPLICATION_M, CUTE_FUSION_SOFTMAX_K64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t tb = {softmax128_b,
                        CUTE_FUSION_SOFTMAX_K64 * sizeof(softmax128_b[0][0]),
                        CUTE_FUSION_SOFTMAX_K64, CUTE_FUSION_N128,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t bias = {softmax128_zero_bias,
                          CUTE_FUSION_N128 * sizeof(softmax128_zero_bias[0][0]),
                          APPLICATION_M, CUTE_FUSION_N128,
                          CUTEDataTypeF16F16F32};

    cute_tiled_matmul_row_block_pipeline_ex(&ta, &tb,
                                            output,
                                            output_stride,
                                            output_elem_bytes,
                                            &bias,
                                            CUTE_TILE_M,
                                            NULL, NULL,
                                            CUTE_SCALE_NONE,
                                            CUTE_BIAS_ZERO, 0,
                                            scratch[0], scratch[1],
                                            post_op, post_ctx);
}

static inline void cute_fusion_run_attention_context_pipeline(
    float output[APPLICATION_M][CUTE_FUSION_CONTEXT_N64])
{
    static float scratch[2][CUTE_TILE_M][CUTE_TILE_N] CUTE_TEST_ALIGN;
    cute_tensor_t ta = {attention_scores,
                        CUTE_FUSION_CONTEXT_K128 * sizeof(attention_scores[0][0]),
                        APPLICATION_M, CUTE_FUSION_CONTEXT_K128,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t tb = {attention_value,
                        CUTE_FUSION_CONTEXT_K128 * sizeof(attention_value[0][0]),
                        CUTE_FUSION_CONTEXT_K128, CUTE_FUSION_CONTEXT_N64,
                        CUTEDataTypeF16F16F32};
    cute_tensor_t bias = {attention_zero_bias,
                          CUTE_FUSION_CONTEXT_N64 * sizeof(attention_zero_bias[0][0]),
                          APPLICATION_M, CUTE_FUSION_CONTEXT_N64,
                          CUTEDataTypeF16F16F32};

    cute_tiled_matmul_pipeline_ex(&ta, &tb,
                                  output,
                                  CUTE_FUSION_CONTEXT_N64 *
                                      sizeof(output[0][0]),
                                  sizeof(output[0][0]),
                                  &bias,
                                  NULL, NULL,
                                  CUTE_SCALE_NONE,
                                  CUTE_BIAS_ZERO, 0,
                                  scratch[0], scratch[1],
                                  NULL, NULL);
}

#endif /* CUTE_FUSION_MATMUL_POST_UTILS_H */
