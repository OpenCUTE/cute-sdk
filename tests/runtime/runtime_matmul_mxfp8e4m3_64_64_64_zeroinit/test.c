#include <stdint.h>
#include "cute_runtime.h"
#include "matmul_value_mxfp8_mnk_64_64_64_zeroinit.h"

int main(void) {
    uint64_t a_stride = STRIDE_A;
    uint64_t b_stride = STRIDE_B;
    uint64_t c_stride = STRIDE_C;
    uint64_t d_stride = STRIDE_D;

    cute_blockscale_matmul(a, a_stride, b, b_stride,
                           a_scale, b_scale,
                           d, d_stride, c, c_stride,
                           APPLICATION_M, APPLICATION_N, APPLICATION_K,
                           ELEMENT_TYPE, BIAS_TYPE, TRANSPOSE_RESULT, 0);

    while (!CUTE_QUERY_MACRO_INST_FINISH())
        ;

    return 0;
}
