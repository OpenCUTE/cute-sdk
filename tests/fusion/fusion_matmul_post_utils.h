#ifndef CUTE_FUSION_MATMUL_POST_UTILS_H
#define CUTE_FUSION_MATMUL_POST_UTILS_H

#include <stdint.h>

#include "cute_fusion.h"
#include "tests/primitive/primitive_test_utils.h"
#include "../runtime/runtime_matmul_i8_128_128_128_zeroinit/matmul_value_mnk_128_128_128_zeroinit.h"

#define CUTE_FUSION_OUTPUT_M APPLICATION_M
#define CUTE_FUSION_OUTPUT_N APPLICATION_N

static float input_scale[APPLICATION_M] CUTE_TEST_ALIGN;
static float weight_scale[1] CUTE_TEST_ALIGN = {0.001f};

static inline float cute_fusion_input_scale_value(int row)
{
    return 0.001f + (float)(row % 17) * 0.00001f;
}

static inline float cute_fusion_residual_value(int row, int col)
{
    int v = ((row * 13 + col * 7) % 257) - 128;
    return (float)v * 0.00025f;
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

#endif /* CUTE_FUSION_MATMUL_POST_UTILS_H */
