#include <stdint.h>

#include "cutelib/primitive/include/cute_convert.h"
#include "golden/manual/vector/dequant_f32_m64_n64/golden_dequant_f32.h"
#include "tests/primitive/primitive_test_utils.h"

static float output[GOLDEN_DEQUANT_M * GOLDEN_DEQUANT_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_dequant_i32_to_f32_tile(golden_dequant_input_i32,
                                 GOLDEN_DEQUANT_N * sizeof(int32_t),
                                 output,
                                 GOLDEN_DEQUANT_N * sizeof(float),
                                 golden_dequant_input_scale,
                                 golden_dequant_weight_scale,
                                 GOLDEN_DEQUANT_M,
                                 GOLDEN_DEQUANT_N);

    cute_test_host_verify_barrier();
    return 0;
}
