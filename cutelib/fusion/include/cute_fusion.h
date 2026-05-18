#ifndef CUTE_FUSION_H
#define CUTE_FUSION_H

#include <stddef.h>
#include <stdint.h>

#include "cute_ops.h"
#include "cute_vector_fusion.h"

static inline const void *cute_fusion_tile_ptr_const(const void *base,
                                                     uint64_t stride,
                                                     int row0,
                                                     int col0,
                                                     size_t elem_bytes)
{
    return (const char *)base + (size_t)row0 * stride + (size_t)col0 * elem_bytes;
}

static inline void cute_post_dequant_rope_bf16cvt(const cute_post_call_t *call)
{
    const cute_rope_ctx_t *base_ctx =
        (const cute_rope_ctx_t *)call->user_ctx;
    cute_rope_ctx_t tile_ctx = {
        .pos = base_ctx->pos + call->tile.row0,
        .rope_theta = base_ctx->rope_theta,
        .key_dim = base_ctx->key_dim,
    };

    cute_fuse_dequant_rope_bf16cvt_tile(
        (const int32_t *)call->tile.src,
        call->tile.src_stride,
        call->tile.dst,
        call->tile.dst_stride,
        call->env.a_scale,
        call->env.b_scale,
        call->tile.rows,
        call->tile.cols,
        &tile_ctx);
}

static inline void cute_post_dequant_bf16cvt(const cute_post_call_t *call)
{
    if (call->env.transpose) {
        cute_fuse_dequant_bf16cvt_transpose_tile(
            (const int32_t *)call->tile.src,
            call->tile.src_stride,
            call->tile.dst,
            call->tile.dst_stride,
            call->env.a_scale,
            call->env.b_scale,
            call->tile.rows,
            call->tile.cols);
    } else {
        cute_fuse_dequant_bf16cvt_tile(
            (const int32_t *)call->tile.src,
            call->tile.src_stride,
            call->tile.dst,
            call->tile.dst_stride,
            call->env.a_scale,
            call->env.b_scale,
            call->tile.rows,
            call->tile.cols);
    }
}

static inline void cute_post_masked_softmax_kvscale_bf16cvt(
    const cute_post_call_t *call)
{
    cute_fuse_masked_softmax_kvscale_bf16cvt_tile(
        (const float *)call->tile.src,
        call->tile.src_stride,
        call->tile.dst,
        call->tile.dst_stride,
        call->tile.row0,
        call->tile.col0,
        call->tile.rows,
        call->tile.cols,
        (const cute_softmax_ctx_t *)call->user_ctx);
}

static inline void cute_post_dequant_silu(const cute_post_call_t *call)
{
    cute_fuse_dequant_silu_tile(
        (const int32_t *)call->tile.src,
        call->tile.src_stride,
        (float *)call->tile.dst,
        call->tile.dst_stride,
        call->env.a_scale,
        call->env.b_scale,
        call->tile.rows,
        call->tile.cols);
}

static inline void cute_post_dequant_hadamard(const cute_post_call_t *call)
{
    const cute_hadamard_ctx_t *base_ctx =
        (const cute_hadamard_ctx_t *)call->user_ctx;
    cute_hadamard_ctx_t tile_ctx = {
        .lhs = (const float *)cute_fusion_tile_ptr_const(
            base_ctx->lhs,
            base_ctx->lhs_stride,
            call->tile.row0,
            call->tile.col0,
            sizeof(float)),
        .lhs_stride = base_ctx->lhs_stride,
        .output_absmax = base_ctx->output_absmax + call->tile.row0,
    };

    cute_fuse_dequant_hadamard_tile(
        (const int32_t *)call->tile.src,
        call->tile.src_stride,
        (float *)call->tile.dst,
        call->tile.dst_stride,
        call->env.a_scale,
        call->env.b_scale,
        call->tile.rows,
        call->tile.cols,
        &tile_ctx);
}

static inline void cute_post_dequant_resadd(const cute_post_call_t *call)
{
    const cute_resadd_ctx_t *base_ctx =
        (const cute_resadd_ctx_t *)call->user_ctx;
    cute_resadd_ctx_t tile_ctx = {
        .residual = (const float *)cute_fusion_tile_ptr_const(
            base_ctx->residual,
            base_ctx->residual_stride,
            call->tile.row0,
            call->tile.col0,
            sizeof(float)),
        .residual_stride = base_ctx->residual_stride,
    };

    cute_fuse_dequant_resadd_tile(
        (const int32_t *)call->tile.src,
        call->tile.src_stride,
        (float *)call->tile.dst,
        call->tile.dst_stride,
        call->env.a_scale,
        call->env.b_scale,
        call->tile.rows,
        call->tile.cols,
        &tile_ctx);
}

#endif /* CUTE_FUSION_H */
