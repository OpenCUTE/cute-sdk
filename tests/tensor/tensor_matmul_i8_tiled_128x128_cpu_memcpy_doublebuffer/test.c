/**
 * Tensor-level tiled matmul test with CPU memcpy post_op.
 * CUTE writes each 64x64 tile to double_buf, then CPU copies it to output.
 * Each output tile is checked with an X-shaped scan against the golden output.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "cute_tensor.h"
#include "cute_ops.h"
#include "../../runtime/runtime_matmul_i8_128_128_128_zeroinit/matmul_value_mnk_128_128_128_zeroinit.h"

static int double_buf[CUTE_TILE_M][CUTE_TILE_N] __attribute__((aligned(256)));

static void cpu_memcpy_post_op(
    void *cute_buf, void *final_out,
    float *a_scale, float *b_scale,
    int dim_i, int dim_j,
    uint64_t cute_stride, uint64_t out_stride,
    void *ctx)
{
    (void)a_scale;
    (void)b_scale;
    (void)ctx;

    size_t row_bytes = (size_t)dim_j * sizeof(int);
    char *src = (char *)cute_buf;
    char *dst = (char *)final_out;

    for (int i = 0; i < dim_i; i++) {
        memcpy(dst + (size_t)i * out_stride,
               src + (size_t)i * cute_stride,
               row_bytes);
    }
}

int main(void) {
    cute_tensor_t ta = {a, APPLICATION_K, APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K, APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t tc = {d, APPLICATION_N * sizeof(int), APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    cute_tiled_matmul(&ta, &tb,
                      c, APPLICATION_N * sizeof(int),
                      &tc,
                      NULL, NULL,               /* no scale */
                      CUTE_SCALE_NONE,
                      BIAS_TYPE, 0,              /* zero init, no transpose */
                      double_buf,                /* CUTE tile output buffer */
                      cpu_memcpy_post_op, NULL); /* CPU copies tile to final output */

    int tile_i = APPLICATION_M / CUTE_TILE_M;
    int tile_j = APPLICATION_N / CUTE_TILE_N;

    for (int ti = 0; ti < tile_i; ti++) {
        for (int tj = 0; tj < tile_j; tj++) {
            int row_base = ti * CUTE_TILE_M;
            int col_base = tj * CUTE_TILE_N;

            for (int k = 0; k < CUTE_TILE_M; k++) {
                int row = row_base + k;
                int col_main = col_base + k;
                int col_anti = col_base + (CUTE_TILE_N - 1 - k);

                if (c[row][col_main] != gloden_c[row][col_main]) {
                    return 1;
                }
                if (c[row][col_anti] != gloden_c[row][col_anti]) {
                    return 1;
                }
            }
        }
    }

    return 0;
}
