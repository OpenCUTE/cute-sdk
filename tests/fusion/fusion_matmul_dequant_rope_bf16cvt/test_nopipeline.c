#include <stdint.h>

#include "../fusion_matmul_post_utils.h"

static float rope_theta[CUTE_FUSION_N64 / 2] CUTE_TEST_ALIGN;
static uint16_t output[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_N64] CUTE_TEST_ALIGN;

int main(void)
{
    cute_rope_ctx_t ctx = {
        .pos = 17,
        .rope_theta = rope_theta,
        .key_dim = CUTE_FUSION_N64,
    };

    cute_fusion_init_scales();
    cute_fusion_init_rope_theta(rope_theta);
    cute_fusion_run_n64_nopipeline(output,
                                   CUTE_FUSION_N64 * sizeof(output[0][0]),
                                   sizeof(output[0][0]),
                                   cute_post_dequant_rope_bf16cvt,
                                   &ctx);
    cute_test_host_verify_barrier();
    return 0;
}
