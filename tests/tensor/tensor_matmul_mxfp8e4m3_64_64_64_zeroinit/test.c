/**
 * Tensor-level blockscale matmul equivalence test.
 */
#include <stdint.h>
#include "cute_tensor.h"
#include "cute_ops.h"
#include "../../runtime/runtime_matmul_mxfp8e4m3_64_64_64_zeroinit/matmul_value_mxfp8_mnk_64_64_64_zeroinit.h"

int main(void) {
    cute_tensor_t ta = {a, APPLICATION_K, APPLICATION_M, APPLICATION_K, ELEMENT_TYPE};
    cute_tensor_t tb = {b, APPLICATION_K, APPLICATION_K, APPLICATION_N, ELEMENT_TYPE};
    cute_tensor_t tc = {d, APPLICATION_N * sizeof(int), APPLICATION_M, APPLICATION_N, ELEMENT_TYPE};
    cute_tensor_t td = {c, APPLICATION_N * sizeof(int), APPLICATION_M, APPLICATION_N, ELEMENT_TYPE};

    uint64_t tid = cute_blockscale_matmul_op(&ta, &tb, &tc, &td,
                                              a_scale, b_scale,
                                              BIAS_TYPE, TRANSPOSE_RESULT, 0);
    cute_wait_task(tid);

    return 0;
}
