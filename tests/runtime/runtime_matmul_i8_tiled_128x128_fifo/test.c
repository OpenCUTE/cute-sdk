/**
 * Tiled matmul FIFO pipeline test
 *
 * Splits 128x128x128 INT8 matmul into 4 tiles (2x2) of 64x64x128,
 * each tile writes directly into its quadrant of c[][] — no CPU copy needed.
 *
 * Pipeline schedule:
 *   1. issue(0,0)
 *   2. issue(0,1)
 *   3. wait(0,0); issue(1,0)
 *   4. wait(0,1); issue(1,1)
 *   5. wait(1,0)
 *   6. wait(1,1)
 */

#include <stddef.h>
#include <stdint.h>
#include "cute_fpe.h"
#include "cute_runtime.h"
#include "matmul_value_mnk_128_128_128_zeroinit.h"

#define TILE_M  64
#define TILE_N  64

int main(void) {
    uint64_t a_stride    = APPLICATION_K;
    uint64_t b_stride    = APPLICATION_K;
    uint64_t out_stride  = APPLICATION_N * sizeof(int);  /* 512 bytes */
    uint64_t bias_stride = APPLICATION_N * sizeof(int);

    int tile_i = APPLICATION_M / TILE_M;
    int tile_j = APPLICATION_N / TILE_N;

    /* ---- issue first two tiles ---- */
    uint64_t tid0 = cute_matmul(
        (const char *)a,                     a_stride,
        (const char *)b,                     b_stride,
        d,                                   bias_stride,
        &c[0][0],                            out_stride,
        TILE_M, TILE_N, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, 0, 0);

    uint64_t tid1 = cute_matmul(
        (const char *)a,                     a_stride,
        (const char *)b + 1 * TILE_N * b_stride, b_stride,
        d,                                   bias_stride,
        &c[0][TILE_N],                       out_stride,
        TILE_M, TILE_N, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, 0, 0);

    /* ---- wait + issue pipeline ---- */
    cute_wait_task(tid0);
    uint64_t tid2 = cute_matmul(
        (const char *)a + 1 * TILE_M * a_stride, a_stride,
        (const char *)b,                         b_stride,
        d,                                       bias_stride,
        &c[TILE_M][0],                           out_stride,
        TILE_M, TILE_N, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, 0, 0);

    cute_wait_task(tid1);
    uint64_t tid3 = cute_matmul(
        (const char *)a + 1 * TILE_M * a_stride, a_stride,
        (const char *)b + 1 * TILE_N * b_stride, b_stride,
        d,                                       bias_stride,
        &c[TILE_M][TILE_N],                      out_stride,
        TILE_M, TILE_N, APPLICATION_K,
        CUTEDataTypeI8I8I32, BIAS_TYPE, 0, 0);

    /* ---- drain ---- */
    cute_wait_task(tid2);
    cute_wait_task(tid3);

    return 0;
}
