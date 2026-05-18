#include <stdint.h>
#include <stdbool.h>

#include "cutelib/primitive/include/cute_quant.h"
#include "golden/manual/vector/smoothquant_m128_k2048/golden_smoothquant.h"
#include "tests/primitive/primitive_test_utils.h"

static int8_t output[GOLDEN_SMOOTHQUANT_M * GOLDEN_SMOOTHQUANT_K] CUTE_TEST_ALIGN;

int main(void)
{
    cute_smoothquant((float *)golden_smoothquant_input,
                     GOLDEN_SMOOTHQUANT_M,
                     GOLDEN_SMOOTHQUANT_K,
                     output,
                     (float *)golden_smoothquant_scale,
                     false);

    cute_test_host_verify_barrier();
    return 0;
}
