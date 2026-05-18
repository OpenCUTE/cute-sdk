#include <stdint.h>

#include "../fusion_matmul_post_utils.h"

static float output[CUTE_FUSION_OUTPUT_M][CUTE_FUSION_CONTEXT_N64] CUTE_TEST_ALIGN;

int main(void)
{
    cute_fusion_init_attention_context_inputs();
    cute_fusion_run_attention_context_nopipeline(output);
    cute_test_host_verify_barrier();
    return 0;
}
