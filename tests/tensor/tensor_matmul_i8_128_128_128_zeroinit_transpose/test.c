/**
 * Tensor-level single matmul with transpose equivalence test.
 */
#include <stdint.h>
#include "cute_tensor.h"
#include "cute_ops.h"
#include "../../runtime/runtime_matmul_i8_128_128_128_zeroinit_transpose/matmul_value_mnk_128_128_128_zeroinit_transpose.h"

int main(void) {
    cute_tensor_t ta = {a, APPLICATION_K, APPLICATION_M, APPLICATION_K, CUTEDataTypeI8I8I32};
    cute_tensor_t tb = {b, APPLICATION_K, APPLICATION_K, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t tc = {d, APPLICATION_N * sizeof(int), APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};
    cute_tensor_t td = {c, APPLICATION_N * sizeof(int), APPLICATION_M, APPLICATION_N, CUTEDataTypeI8I8I32};

    uint64_t tid = cute_matmul_op(&ta, &tb, &tc, &td, BIAS_TYPE, TRANSPOSE_RESULT, 0);
    cute_wait_task(tid);

    return 0;
}
