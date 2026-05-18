#include <stdint.h>

#include "../fusion_matmul_post_utils.h"

static uint16_t output[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_OUTPUT_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_fusion_init_scales();
    cute_fusion_run_pipeline(output,
                             CUTE_FUSION_OUTPUT_N * sizeof(output[0][0]),
                             sizeof(output[0][0]),
                             cute_post_dequant_bf16cvt,
                             NULL);
    cute_test_host_verify_barrier();
    return 0;
}
