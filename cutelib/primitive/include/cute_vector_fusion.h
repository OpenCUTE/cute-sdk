#ifndef CUTE_VECTOR_FUSION_H
#define CUTE_VECTOR_FUSION_H

#include <stddef.h>
#include <stdint.h>

#include "cute_convert.h"
#include "cute_elementwise.h"
#include "cute_sequence.h"

typedef struct {
    int pos;
    const float *rope_theta;
    int key_dim;
} cute_rope_ctx_t;

typedef struct {
    int pos;
    const int8_t *bitmask;
    int max_ctx_len;
    float kv_scale;
} cute_softmax_ctx_t;

typedef struct {
    const float *residual;
    uint64_t residual_stride;
} cute_resadd_ctx_t;

typedef struct {
    const float *lhs;
    uint64_t lhs_stride;
    float *output_absmax;
} cute_hadamard_ctx_t;

static inline void cute_fuse_dequant_rope_bf16cvt_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols,
    const cute_rope_ctx_t *ctx)
{
    float dequant[rows * cols];

    cute_dequant_i32_to_f32_tile(input, input_stride,
                                 dequant, (uint64_t)cols * sizeof(float),
                                 input_scale, weight_scale,
                                 rows, cols);
    cute_rope_bf16_tile(dequant, (uint64_t)cols * sizeof(float),
                        output, output_stride,
                        ctx->rope_theta,
                        ctx->pos,
                        ctx->key_dim,
                        rows, cols);
}

static inline void cute_fuse_dequant_bf16cvt_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols)
{
    cute_dequant_i32_to_bf16_tile(input, input_stride,
                                  output, output_stride,
                                  input_scale, weight_scale,
                                  rows, cols);
}

static inline void cute_fuse_dequant_bf16cvt_transpose_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols)
{
    cute_dequant_i32_to_bf16_transpose_tile(input, input_stride,
                                            output, output_stride,
                                            input_scale, weight_scale,
                                            rows, cols);
}

static inline void cute_fuse_masked_softmax_kvscale_bf16cvt_tile(
    const float *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    int row0, int col0,
    int rows, int cols,
    const cute_softmax_ctx_t *ctx)
{
    const uint64_t mask_stride = (uint64_t)(ctx->max_ctx_len + 7) / 8;
    cute_masked_softmax_bf16_tile(input, input_stride,
                                  output, output_stride,
                                  ctx->bitmask, mask_stride,
                                  ctx->kv_scale,
                                  row0 + ctx->pos,
                                  col0,
                                  rows, cols);
}

static inline void cute_fuse_dequant_silu_tile(
    const int32_t *input, uint64_t input_stride,
    float *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols)
{
    cute_dequant_i32_to_f32_tile(input, input_stride,
                                 output, output_stride,
                                 input_scale, weight_scale,
                                 rows, cols);
    cute_silu_tile_fast(output, output_stride, rows, cols);
}

static inline void cute_fuse_dequant_hadamard_tile(
    const int32_t *input, uint64_t input_stride,
    float *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols,
    const cute_hadamard_ctx_t *ctx)
{
    cute_dequant_i32_to_f32_tile(input, input_stride,
                                 output, output_stride,
                                 input_scale, weight_scale,
                                 rows, cols);
    cute_hadamard_tile(ctx->lhs, ctx->lhs_stride,
                       output, output_stride,
                       output, output_stride,
                       ctx->output_absmax,
                       rows, cols);
}

static inline void cute_fuse_dequant_resadd_tile(
    const int32_t *input, uint64_t input_stride,
    float *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols,
    const cute_resadd_ctx_t *ctx)
{
    cute_dequant_i32_to_f32_tile(input, input_stride,
                                 output, output_stride,
                                 input_scale, weight_scale,
                                 rows, cols);
    cute_resadd_tile(output, output_stride,
                     ctx->residual, ctx->residual_stride,
                     output, output_stride,
                     rows, cols);
}

#endif /* CUTE_VECTOR_FUSION_H */
