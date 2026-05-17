#include <stdint.h>

#include "cutelib/primitive/include/cute_quant.h"
#include "golden/manual/vector/rmsnorm_batch1_seq_len128_hidden_dim2048/golden_rmsnorm.h"
#include "tests/primitive/primitive_test_utils.h"

static float output[GOLDEN_RMSNORM_BATCH * GOLDEN_RMSNORM_SEQ_LEN * GOLDEN_RMSNORM_HIDDEN_DIM] CUTE_TEST_ALIGN;

int main(void)
{
    cute_rmsnorm(golden_rmsnorm_input,
                 output,
                 golden_rmsnorm_weight,
                 1e-5f,
                 GOLDEN_RMSNORM_BATCH,
                 GOLDEN_RMSNORM_SEQ_LEN,
                 GOLDEN_RMSNORM_HIDDEN_DIM);

    cute_test_host_verify_barrier();
    return 0;
}
