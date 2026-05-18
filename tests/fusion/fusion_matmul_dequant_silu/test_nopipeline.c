#include "../fusion_matmul_post_utils.h"

static float output[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_OUTPUT_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_fusion_init_scales();
    cute_fusion_run_nopipeline(output,
                               CUTE_FUSION_OUTPUT_N * sizeof(output[0][0]),
                               sizeof(output[0][0]),
                               cute_post_dequant_silu,
                               NULL);
    cute_test_host_verify_barrier();
    return 0;
}
