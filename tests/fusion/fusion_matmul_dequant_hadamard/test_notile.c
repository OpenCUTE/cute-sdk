#include "../fusion_matmul_post_utils.h"

static float lhs[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_OUTPUT_N] CUTE_TEST_ALIGN;
static float output[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_OUTPUT_N] CUTE_TEST_ALIGN;
static float row_absmax[CUTE_FUSION_OUTPUT_M] CUTE_TEST_ALIGN;

int main(void)
{
    cute_hadamard_ctx_t ctx = {
        .lhs = &lhs[0][0],
        .lhs_stride = CUTE_FUSION_OUTPUT_N * sizeof(lhs[0][0]),
        .output_absmax = row_absmax,
    };

    cute_fusion_init_scales();
    cute_fusion_init_hadamard_lhs(lhs);
    cute_fusion_zero_f32(row_absmax, CUTE_FUSION_OUTPUT_M);
    cute_fusion_run_notile(output,
                           CUTE_FUSION_OUTPUT_N * sizeof(output[0][0]),
                           cute_post_dequant_hadamard,
                           &ctx);
    cute_test_publish_f32_trace(row_absmax, CUTE_FUSION_OUTPUT_M);
    cute_test_host_verify_barrier();
    return 0;
}
