#include <stdint.h>
#include "cute_fpe.h"
#include "cute_runtime.h"
#include "matmul_value_mnk_128_128_128_zeroinit.h"

int main(void) {
    uint64_t a_stride = APPLICATION_K * sizeof(a[0][0]);
    uint64_t b_stride = APPLICATION_K * sizeof(b[0][0]);
    uint64_t c_stride = APPLICATION_N * sizeof(c[0][0]);
    uint64_t d_stride = APPLICATION_N * sizeof(d[0][0]);

    cute_matmul(a, a_stride, b, b_stride,
                d, d_stride, c, c_stride,
                APPLICATION_M, APPLICATION_N, APPLICATION_K,
                CUTEDataTypeI8I8I32, BIAS_TYPE, TRANSPOSE_RESULT, 0);

    while (!CUTE_QUERY_MACRO_INST_FINISH())
        ;

    return 0;
}
