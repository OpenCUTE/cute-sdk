#include <stdint.h>

#include "cutelib/primitive/include/cute_convert.h"
#include "golden/manual/vector/dequant_f16_m64_n64/golden_dequant_f16.h"
#include "tests/primitive/primitive_test_utils.h"

static uint16_t output[GOLDEN_DEQUANT_M * GOLDEN_DEQUANT_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_dequant_i32_to_f16_tile(golden_dequant_input_i32,
                                 GOLDEN_DEQUANT_N * sizeof(int32_t),
                                 output,
                                 GOLDEN_DEQUANT_N * sizeof(uint16_t),
                                 golden_dequant_input_scale,
                                 golden_dequant_weight_scale,
                                 GOLDEN_DEQUANT_M,
                                 GOLDEN_DEQUANT_N);

    return 0;
}
