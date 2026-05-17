#include <stdint.h>

#include "cutelib/primitive/include/cute_sequence.h"
#include "golden/manual/vector/rope_pos0_m64_head_dim64_n_head1/golden_rope_pos0.h"
#include "tests/primitive/primitive_test_utils.h"

static uint16_t output[GOLDEN_ROPE_M * GOLDEN_ROPE_HEAD_DIM] CUTE_TEST_ALIGN;

int main(void)
{
    cute_rope_f16_tile(golden_rope_input,
                       GOLDEN_ROPE_HEAD_DIM * sizeof(float),
                       output,
                       GOLDEN_ROPE_HEAD_DIM * sizeof(uint16_t),
                       golden_rope_theta,
                       GOLDEN_ROPE_POS,
                       GOLDEN_ROPE_HEAD_DIM,
                       GOLDEN_ROPE_M,
                       GOLDEN_ROPE_HEAD_DIM);

    return cute_check_star64_blocks(output,
                                    golden_rope_output_f16,
                                    GOLDEN_ROPE_M,
                                    GOLDEN_ROPE_HEAD_DIM,
                                    sizeof(output[0]));
}
