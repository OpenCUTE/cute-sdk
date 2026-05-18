#include <stdint.h>

#include "cutelib/primitive/include/cute_vector_fusion.h"
#include "golden/manual/vector/fuse_dequant_resadd_m64_n64/golden_fuse_dequant_resadd.h"
#include "tests/primitive/primitive_test_utils.h"

static float output[GOLDEN_FUSE_TOTAL] CUTE_TEST_ALIGN;

int main(void)
{
    const cute_resadd_ctx_t ctx = {
        .residual = golden_fuse_residual,
        .residual_stride = GOLDEN_FUSE_N * sizeof(float),
    };

    cute_fuse_dequant_resadd_tile(golden_fuse_input_i32,
                                  GOLDEN_FUSE_N * sizeof(int32_t),
                                  output,
                                  GOLDEN_FUSE_N * sizeof(float),
                                  golden_fuse_input_scale,
                                  golden_fuse_weight_scale,
                                  GOLDEN_FUSE_M,
                                  GOLDEN_FUSE_N,
                                  &ctx);

    cute_test_host_verify_barrier();
    return 0;
}
