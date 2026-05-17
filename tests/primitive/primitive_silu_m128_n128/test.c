#include <stdint.h>

#include "cutelib/primitive/include/cute_elementwise.h"
#include "golden/manual/vector/silu_m128_n128/golden_silu.h"
#include "tests/primitive/primitive_test_utils.h"

static float data[GOLDEN_SILU_TOTAL] CUTE_TEST_ALIGN;

int main(void)
{
    memcpy(data, golden_silu_input_x, sizeof(data));
    cute_silu_tile(data,
                   GOLDEN_SILU_N * sizeof(float),
                   GOLDEN_SILU_M,
                   GOLDEN_SILU_N);

    cute_test_host_verify_barrier();
    return 0;
}
