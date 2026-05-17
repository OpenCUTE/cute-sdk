#include <stdint.h>

#include "cutelib/primitive/include/cute_elementwise.h"
#include "golden/manual/vector/resadd_m64_n64/golden_resadd.h"
#include "tests/primitive/primitive_test_utils.h"

static float output[GOLDEN_RESADD_M * GOLDEN_RESADD_N] CUTE_TEST_ALIGN;

int main(void)
{
    cute_resadd_tile(golden_resadd_lhs,
                     GOLDEN_RESADD_N * sizeof(float),
                     golden_resadd_rhs,
                     GOLDEN_RESADD_N * sizeof(float),
                     output,
                     GOLDEN_RESADD_N * sizeof(float),
                     GOLDEN_RESADD_M,
                     GOLDEN_RESADD_N);

    return cute_check_star64_blocks(output,
                                    golden_resadd_output,
                                    GOLDEN_RESADD_M,
                                    GOLDEN_RESADD_N,
                                    sizeof(output[0]));
}
