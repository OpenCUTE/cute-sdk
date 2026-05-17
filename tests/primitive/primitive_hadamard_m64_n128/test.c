#include <stdint.h>

#include "cutelib/primitive/include/cute_elementwise.h"
#include "golden/manual/vector/hadamard_m64_n128/golden_hadamard.h"
#include "tests/primitive/primitive_test_utils.h"

static float output[GOLDEN_HADAMARD_M * GOLDEN_HADAMARD_N] CUTE_TEST_ALIGN;
static float row_absmax[GOLDEN_HADAMARD_M] CUTE_TEST_ALIGN;

int main(void)
{
    const int split_cols = GOLDEN_HADAMARD_N / 2;
    cute_hadamard_tile(golden_hadamard_lhs,
                       GOLDEN_HADAMARD_N * sizeof(float),
                       golden_hadamard_rhs,
                       GOLDEN_HADAMARD_N * sizeof(float),
                       output,
                       GOLDEN_HADAMARD_N * sizeof(float),
                       row_absmax,
                       GOLDEN_HADAMARD_M,
                       split_cols);
    cute_hadamard_tile(golden_hadamard_lhs + split_cols,
                       GOLDEN_HADAMARD_N * sizeof(float),
                       golden_hadamard_rhs + split_cols,
                       GOLDEN_HADAMARD_N * sizeof(float),
                       output + split_cols,
                       GOLDEN_HADAMARD_N * sizeof(float),
                       row_absmax,
                       GOLDEN_HADAMARD_M,
                       GOLDEN_HADAMARD_N - split_cols);

    return 0;
}
