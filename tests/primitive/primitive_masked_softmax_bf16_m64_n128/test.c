#include <stdint.h>

#include "cutelib/primitive/include/cute_sequence.h"
#include "golden/manual/vector/masked_softmax_bf16_m64_n128/golden_masked_softmax_bf16.h"
#include "tests/primitive/primitive_test_utils.h"

static uint16_t output[GOLDEN_SOFTMAX_M * GOLDEN_SOFTMAX_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_masked_softmax_bf16_tile(golden_softmax_input,
                                  GOLDEN_SOFTMAX_N * sizeof(float),
                                  output,
                                  GOLDEN_SOFTMAX_N * sizeof(uint16_t),
                                  (const int8_t *)golden_softmax_causal_mask,
                                  (GOLDEN_SOFTMAX_N + 7) / 8,
                                  1.0f,
                                  0,
                                  0,
                                  GOLDEN_SOFTMAX_M,
                                  GOLDEN_SOFTMAX_N);

    cute_test_host_verify_barrier();
    return 0;
}
