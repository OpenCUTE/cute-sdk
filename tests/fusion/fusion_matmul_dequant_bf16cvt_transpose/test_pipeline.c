#include <stdint.h>

#include "../fusion_matmul_post_utils.h"

static uint16_t output[CUTE_FUSION_OUTPUT_N][CUTE_FUSION_OUTPUT_M] CUTE_TEST_ALIGN;

int main(void)
{
    cute_fusion_init_scales();
    cute_fusion_run_bf16cvt_transpose_pipeline(
        output,
        CUTE_FUSION_OUTPUT_M * sizeof(output[0][0]),
        sizeof(output[0][0]));
    cute_test_host_verify_barrier();
    return 0;
}
