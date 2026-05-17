#include <stdint.h>

#include "cutelib/primitive/include/cute_sequence.h"
#include "golden/manual/vector/masked_softmax_m64_n128/golden_masked_softmax.h"
#include "tests/primitive/primitive_test_utils.h"

static uint16_t output[GOLDEN_SOFTMAX_M * GOLDEN_SOFTMAX_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_masked_softmax_f16_tile(golden_softmax_input,
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

    return cute_check_star64_blocks(output,
                                    golden_softmax_output_f16,
                                    GOLDEN_SOFTMAX_M,
                                    GOLDEN_SOFTMAX_N,
                                    sizeof(output[0]));
}
