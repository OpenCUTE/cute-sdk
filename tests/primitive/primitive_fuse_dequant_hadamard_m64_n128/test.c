#include <stdint.h>

#include "cutelib/primitive/include/cute_vector_fusion.h"
#include "golden/manual/vector/fuse_dequant_hadamard_m64_n128/golden_fuse_dequant_hadamard.h"
#include "tests/primitive/primitive_test_utils.h"

static float output[GOLDEN_FUSE_TOTAL] CUTE_TEST_ALIGN;
static float row_absmax[GOLDEN_FUSE_M] CUTE_TEST_ALIGN;

int main(void)
{
    const int split_cols = GOLDEN_FUSE_N / 2;
    cute_hadamard_ctx_t ctx = {
        .lhs = golden_fuse_lhs,
        .lhs_stride = GOLDEN_FUSE_N * sizeof(float),
        .output_absmax = row_absmax,
    };

    cute_fuse_dequant_hadamard_tile(golden_fuse_input_i32,
                                    GOLDEN_FUSE_N * sizeof(int32_t),
                                    output,
                                    GOLDEN_FUSE_N * sizeof(float),
                                    golden_fuse_input_scale,
                                    golden_fuse_weight_scale,
                                    GOLDEN_FUSE_M,
                                    split_cols,
                                    &ctx);

    ctx.lhs = golden_fuse_lhs + split_cols;
    cute_fuse_dequant_hadamard_tile(golden_fuse_input_i32 + split_cols,
                                    GOLDEN_FUSE_N * sizeof(int32_t),
                                    output + split_cols,
                                    GOLDEN_FUSE_N * sizeof(float),
                                    golden_fuse_input_scale,
                                    golden_fuse_weight_scale,
                                    GOLDEN_FUSE_M,
                                    GOLDEN_FUSE_N - split_cols,
                                    &ctx);

    cute_test_publish_f32_trace(row_absmax, GOLDEN_FUSE_M);
    cute_test_host_verify_barrier();
    return 0;
}
