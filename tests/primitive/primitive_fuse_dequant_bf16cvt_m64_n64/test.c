#include <stdint.h>

#include "cutelib/primitive/include/cute_vector_fusion.h"
#include "golden/manual/vector/fuse_dequant_bf16cvt_m64_n64/golden_fuse_dequant_bf16cvt.h"
#include "tests/primitive/primitive_test_utils.h"

static uint16_t output[GOLDEN_FUSE_TOTAL] CUTE_TEST_ALIGN;

int main(void)
{
    cute_fuse_dequant_bf16cvt_tile(golden_fuse_input_i32,
                                   GOLDEN_FUSE_N * sizeof(int32_t),
                                   output,
                                   GOLDEN_FUSE_N * sizeof(uint16_t),
                                   golden_fuse_input_scale,
                                   golden_fuse_weight_scale,
                                   GOLDEN_FUSE_M,
                                   GOLDEN_FUSE_N);

    cute_test_host_verify_barrier();
    return 0;
}
