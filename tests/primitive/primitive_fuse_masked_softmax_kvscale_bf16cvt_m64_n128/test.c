#include <stdint.h>

#include "cutelib/primitive/include/cute_vector_fusion.h"
#include "golden/manual/vector/fuse_masked_softmax_kvscale_bf16cvt_m64_n128/golden_fuse_masked_softmax_kvscale_bf16cvt.h"
#include "tests/primitive/primitive_test_utils.h"

static uint16_t output[GOLDEN_FUSE_TOTAL] CUTE_TEST_ALIGN;

int main(void)
{
    const cute_softmax_ctx_t ctx = {
        .pos = 0,
        .bitmask = (const int8_t *)golden_fuse_causal_mask,
        .max_ctx_len = GOLDEN_FUSE_N,
        .kv_scale = GOLDEN_FUSE_KVSCALE,
    };

    cute_fuse_masked_softmax_kvscale_bf16cvt_tile(golden_fuse_input,
                                                  GOLDEN_FUSE_N * sizeof(float),
                                                  output,
                                                  GOLDEN_FUSE_N * sizeof(uint16_t),
                                                  0,
                                                  0,
                                                  GOLDEN_FUSE_M,
                                                  GOLDEN_FUSE_N,
                                                  &ctx);

    cute_test_host_verify_barrier();
    return 0;
}
