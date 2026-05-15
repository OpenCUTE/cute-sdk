#ifndef CUTE_OPS_H
#define CUTE_OPS_H

#include <stddef.h>
#include "cute_tensor.h"
#include "cute_runtime.h"

/* ---- 单 tile matmul ---- */

/* D = A x B + C，直接映射到 cute_matmul */
static inline uint64_t cute_matmul_op(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    const cute_tensor_t *bias,
    const cute_tensor_t *output,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t m_index)
{
    return cute_matmul(
        a->data, a->stride,
        b->data, b->stride,
        bias->data, bias->stride,
        output->data, output->stride,
        a->rows, output->cols, a->cols,
        a->dtype, bias_mode, transpose, m_index);
}

/* Blockscale 变体 */
static inline uint64_t cute_blockscale_matmul_op(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    const cute_tensor_t *bias,
    const cute_tensor_t *output,
    const void *scale_a, const void *scale_b,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t m_index)
{
    return cute_blockscale_matmul(
        a->data, a->stride,
        b->data, b->stride,
        scale_a, scale_b,
        bias->data, bias->stride,
        output->data, output->stride,
        a->rows, output->cols, a->cols,
        a->dtype, bias_mode, transpose, m_index);
}

/* ---- Post-op 回调类型 ---- */
typedef struct {
    void *src;
    void *dst;
    uint64_t src_stride;
    uint64_t dst_stride;
    int rows;
    int cols;
    int tile_i;
    int tile_j;
    int row0;
    int col0;
} cute_post_tile_t;

typedef struct {
    float *a_scale;
    float *b_scale;
    int scale_type;
    int bias_mode;
    int transpose;
} cute_post_env_t;

typedef struct {
    cute_post_tile_t tile;
    cute_post_env_t env;
    void *user_ctx;
} cute_post_call_t;

typedef void (*cute_post_op_fn)(const cute_post_call_t *call);

static inline void cute_run_post_op(
    cute_post_op_fn post_op,
    void *src,
    void *dst,
    uint64_t src_stride,
    uint64_t dst_stride,
    float *a_scale,
    float *b_scale,
    int scale_type,
    int bias_mode,
    int transpose,
    int tile_i,
    int tile_j,
    void *user_ctx)
{
    cute_post_call_t call;
    call.tile.src = src;
    call.tile.dst = dst;
    call.tile.src_stride = src_stride;
    call.tile.dst_stride = dst_stride;
    call.tile.rows = CUTE_TILE_M;
    call.tile.cols = CUTE_TILE_N;
    call.tile.tile_i = tile_i;
    call.tile.tile_j = tile_j;
    call.tile.row0 = tile_i * CUTE_TILE_M;
    call.tile.col0 = tile_j * CUTE_TILE_N;
    call.env.a_scale = a_scale ? a_scale + call.tile.row0 : NULL;
    call.env.b_scale = b_scale;
    call.env.scale_type = scale_type;
    call.env.bias_mode = bias_mode;
    call.env.transpose = transpose;
    call.user_ctx = user_ctx;
    post_op(&call);
}

/* ---- Tiled Matmul without post-op pipeline ---- */
static inline void cute_tiled_matmul_no_pipeline(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    void *output,
    uint64_t output_stride,
    const cute_tensor_t *bias,
    float *a_scale, float *b_scale,
    int scale_type, int bias_mode, int transpose,
    void *double_buf,
    cute_post_op_fn post_op,
    void *post_ctx)
{
    int M = (int)a->rows, N = (int)b->cols, K = (int)a->cols;
    int tile_i = M / CUTE_TILE_M;
    int tile_j = N / CUTE_TILE_N;
    int total = tile_i * tile_j;
    uint64_t tile_out_stride = CUTE_TILE_N * 4;

    if (total == 0)
        return;

    /* helper: compute tile output pointer */
    #define _TILE_OUT_PTR(ti, tj) \
        ((char *)output + (ti) * CUTE_TILE_M * output_stride + (tj) * CUTE_TILE_N * 4)

    /* helper: compute tile A data pointer */
    #define _TILE_A_PTR(ti) \
        ((char *)a->data + (ti) * CUTE_TILE_M * a->stride)

    /* helper: compute tile B data pointer */
    #define _TILE_B_PTR(tj) \
        ((char *)b->data + (tj) * CUTE_TILE_N * b->stride)

    if (post_op == NULL) {
        /* --- Direct-write path: CUTE writes directly to output quadrant --- */

        /* Issue first tile */
        int ti0 = 0, tj0 = 0;
        uint64_t tid = cute_matmul(
            _TILE_A_PTR(ti0), a->stride,
            _TILE_B_PTR(tj0), b->stride,
            bias->data, bias->stride,
            _TILE_OUT_PTR(ti0, tj0), output_stride,
            CUTE_TILE_M, CUTE_TILE_N, K,
            a->dtype, bias_mode, transpose, 0);

        /* Wait previous tile before issuing the next one. */
        for (int n = 1; n < total; n++) {
            int ti = n / tile_j, tj = n % tile_j;

            cute_wait_task(tid);
            tid = cute_matmul(
                _TILE_A_PTR(ti), a->stride,
                _TILE_B_PTR(tj), b->stride,
                bias->data, bias->stride,
                _TILE_OUT_PTR(ti, tj), output_stride,
                CUTE_TILE_M, CUTE_TILE_N, K,
                a->dtype, bias_mode, transpose, 0);
        }

        /* Drain last */
        cute_wait_task(tid);
    } else {
        /* --- Post_op path: CUTE writes to double_buf, CPU processes --- */

        /* Issue first tile */
        int ti0 = 0, tj0 = 0;
        uint64_t tid = cute_matmul(
            _TILE_A_PTR(ti0), a->stride,
            _TILE_B_PTR(tj0), b->stride,
            bias->data, bias->stride,
            double_buf, tile_out_stride,
            CUTE_TILE_M, CUTE_TILE_N, K,
            a->dtype, bias_mode, transpose, 0);
        int prev_ti = ti0, prev_tj = tj0;

        /* With one scratch tile, CPU must consume it before CUTE reuses it. */
        for (int n = 1; n < total; n++) {
            int ti = n / tile_j, tj = n % tile_j;

            cute_wait_task(tid);

            cute_run_post_op(post_op, double_buf, _TILE_OUT_PTR(prev_ti, prev_tj),
                             tile_out_stride, output_stride,
                             a_scale, b_scale, scale_type, bias_mode, transpose,
                             prev_ti, prev_tj, post_ctx);

            /* Issue current tile into double_buf after CPU has copied prev. */
            tid = cute_matmul(
                _TILE_A_PTR(ti), a->stride,
                _TILE_B_PTR(tj), b->stride,
                bias->data, bias->stride,
                double_buf, tile_out_stride,
                CUTE_TILE_M, CUTE_TILE_N, K,
                a->dtype, bias_mode, transpose, 0);
            prev_ti = ti; prev_tj = tj;
        }

        /* Drain last tile */
        cute_wait_task(tid);
        cute_run_post_op(post_op, double_buf, _TILE_OUT_PTR(prev_ti, prev_tj),
                         tile_out_stride, output_stride,
                         a_scale, b_scale, scale_type, bias_mode, transpose,
                         prev_ti, prev_tj, post_ctx);
    }

    #undef _TILE_OUT_PTR
    #undef _TILE_A_PTR
    #undef _TILE_B_PTR
}

/* ---- Tiled Matmul with two-buffer post-op pipeline ---- */
static inline void cute_tiled_matmul_pipeline(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    void *output,
    uint64_t output_stride,
    const cute_tensor_t *bias,
    float *a_scale, float *b_scale,
    int scale_type, int bias_mode, int transpose,
    void *double_buf0, void *double_buf1,
    cute_post_op_fn post_op,
    void *post_ctx)
{
    int M = (int)a->rows, N = (int)b->cols, K = (int)a->cols;
    int tile_i = M / CUTE_TILE_M;
    int tile_j = N / CUTE_TILE_N;
    int total = tile_i * tile_j;
    uint64_t tile_out_stride = CUTE_TILE_N * 4;
    void *bufs[2] = {double_buf0, double_buf1};

    if (post_op == NULL || double_buf0 == NULL || double_buf1 == NULL) {
        cute_tiled_matmul_no_pipeline(a, b, output, output_stride, bias,
                                      a_scale, b_scale, scale_type,
                                      bias_mode, transpose,
                                      double_buf0, post_op, post_ctx);
        return;
    }

    if (total == 0)
        return;

    #define _TILE_OUT_PTR(ti, tj) \
        ((char *)output + (ti) * CUTE_TILE_M * output_stride + (tj) * CUTE_TILE_N * 4)

    #define _TILE_A_PTR(ti) \
        ((char *)a->data + (ti) * CUTE_TILE_M * a->stride)

    #define _TILE_B_PTR(tj) \
        ((char *)b->data + (tj) * CUTE_TILE_N * b->stride)

    int prev_ti = 0, prev_tj = 0, prev_buf = 0;
    uint64_t tid = cute_matmul(
        _TILE_A_PTR(prev_ti), a->stride,
        _TILE_B_PTR(prev_tj), b->stride,
        bias->data, bias->stride,
        bufs[prev_buf], tile_out_stride,
        CUTE_TILE_M, CUTE_TILE_N, K,
        a->dtype, bias_mode, transpose, 0);

    for (int n = 1; n < total; n++) {
        int ti = n / tile_j, tj = n % tile_j;
        int curr_buf = n & 1;

        cute_wait_task(tid);

        tid = cute_matmul(
            _TILE_A_PTR(ti), a->stride,
            _TILE_B_PTR(tj), b->stride,
            bias->data, bias->stride,
            bufs[curr_buf], tile_out_stride,
            CUTE_TILE_M, CUTE_TILE_N, K,
            a->dtype, bias_mode, transpose, 0);

        cute_run_post_op(post_op, bufs[prev_buf], _TILE_OUT_PTR(prev_ti, prev_tj),
                         tile_out_stride, output_stride,
                         a_scale, b_scale, scale_type, bias_mode, transpose,
                         prev_ti, prev_tj, post_ctx);

        prev_ti = ti;
        prev_tj = tj;
        prev_buf = curr_buf;
    }

    cute_wait_task(tid);
    cute_run_post_op(post_op, bufs[prev_buf], _TILE_OUT_PTR(prev_ti, prev_tj),
                     tile_out_stride, output_stride,
                     a_scale, b_scale, scale_type, bias_mode, transpose,
                     prev_ti, prev_tj, post_ctx);

    #undef _TILE_OUT_PTR
    #undef _TILE_A_PTR
    #undef _TILE_B_PTR
}

/* Backward-compatible alias for the non-pipelined implementation. */
static inline void cute_tiled_matmul(
    const cute_tensor_t *a,
    const cute_tensor_t *b,
    void *output,
    uint64_t output_stride,
    const cute_tensor_t *bias,
    float *a_scale, float *b_scale,
    int scale_type, int bias_mode, int transpose,
    void *double_buf,
    cute_post_op_fn post_op,
    void *post_ctx)
{
    cute_tiled_matmul_no_pipeline(a, b, output, output_stride, bias,
                                  a_scale, b_scale, scale_type,
                                  bias_mode, transpose,
                                  double_buf, post_op, post_ctx);
}

#endif /* CUTE_OPS_H */
