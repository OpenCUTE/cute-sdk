#include <stdint.h>

#include "../fusion_matmul_post_utils.h"

static int8_t causal_mask[CUTE_FUSION_OUTPUT_M * ((CUTE_FUSION_N64 + 7) / 8)]
    CUTE_TEST_ALIGN;
static uint16_t output[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_N64] CUTE_TEST_ALIGN;

int main(void)
{
    cute_softmax_ctx_t ctx = {
        .pos = 0,
        .bitmask = causal_mask,
        .max_ctx_len = CUTE_FUSION_N64,
        .kv_scale = 0.125f,
    };

    cute_fusion_init_softmax_inputs();
    cute_fusion_init_causal_mask(causal_mask,
                                 CUTE_FUSION_OUTPUT_M,
                                 CUTE_FUSION_N64);
    cute_fusion_run_softmax_pipeline(output,
                                     CUTE_FUSION_N64 * sizeof(output[0][0]),
                                     sizeof(output[0][0]),
                                     cute_post_masked_softmax_kvscale_bf16cvt,
                                     &ctx);
    cute_test_host_verify_barrier();
    return 0;
}
