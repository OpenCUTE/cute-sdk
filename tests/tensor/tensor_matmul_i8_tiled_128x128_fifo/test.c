/**
 * Tensor-level tiled matmul test.
 * Uses cute_tiled_matmul_no_pipeline to split 128x128x128 into 2x2 tiles of 64x64x128.
 * post_op=NULL: CUTE writes directly to output quadrants.
 * Result must be bit-exact with single-call 128x128 matmul.
 */
#include <stddef.h>
#include <stdint.h>
#include "cute_tensor.h"
#include "cute_ops.h"
#include "../../runtime/runtime_matmul_i8_128_128_128_zeroinit/matmul_value_mnk_128_128_128_zeroinit.h"

int main(void) {
    cute_tensor_t ta = {a, APPLICATION_K, APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K, APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t tc = {d, APPLICATION_N * sizeof(int), APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    cute_tiled_matmul_no_pipeline(&ta, &tb,
                                  c, APPLICATION_N * sizeof(int),
                                  &tc,
                                  NULL, NULL,               /* no scale */
                                  CUTE_SCALE_NONE,
                                  BIAS_TYPE, 0,              /* zero init, no transpose */
                                  NULL,                      /* no double_buf needed (post_op=NULL) */
                                  NULL, NULL);               /* no post_op */

    return 0;
}
