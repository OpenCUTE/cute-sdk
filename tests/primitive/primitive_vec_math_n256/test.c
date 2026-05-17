#include <stdint.h>
#include <riscv_vector.h>

#include "cutelib/primitive/include/cute_vec_math.h"
#include "golden/manual/vector/vec_math_n256/golden_vec_math.h"
#include "tests/primitive/primitive_test_utils.h"

static float out_exp[GOLDEN_VEC_MATH_N] CUTE_TEST_ALIGN;
static float out_sin[GOLDEN_VEC_MATH_N] CUTE_TEST_ALIGN;
static float out_cos[GOLDEN_VEC_MATH_N] CUTE_TEST_ALIGN;

int main(void)
{
    size_t vl;
    for (int i = 0, avl = GOLDEN_VEC_MATH_N; avl > 0; i += vl, avl -= vl) {
        vl = __riscv_vsetvl_e32m4(avl);
        vfloat32m4_t x = __riscv_vle32_v_f32m4(&golden_vec_math_input_x[i], vl);
        __riscv_vse32_v_f32m4(&out_exp[i], cute_vec_exp(x, vl), vl);
        __riscv_vse32_v_f32m4(&out_sin[i], cute_vec_sin(x, vl), vl);
        __riscv_vse32_v_f32m4(&out_cos[i], cute_vec_cos(x, vl), vl);
    }

    return 0;
}
