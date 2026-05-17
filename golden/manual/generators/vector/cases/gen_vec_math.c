/*
 * Golden generator for vec_math: exp, sin, cos.
 * Params: GOLDEN_N
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_gloden_opt.h"

#ifndef GOLDEN_N
#define GOLDEN_N 256
#endif

static void print_f32_array(const char *name, const float *data, int n) {
    printf("static const float %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 4 == 0) printf("    ");
        printf("%a", data[i]);
        if (i < n - 1) printf(", ");
        if (i % 4 == 3 || i == n - 1) printf("\n");
    }
    printf("};\n");
}

int main(void) {
    float input[GOLDEN_N];
    float out_exp[GOLDEN_N], out_sin[GOLDEN_N], out_cos[GOLDEN_N];

    memset(input, 0, sizeof(input));
    memset(out_exp, 0, sizeof(out_exp));
    memset(out_sin, 0, sizeof(out_sin));
    memset(out_cos, 0, sizeof(out_cos));

    cute_fill_vec_math_input(input, GOLDEN_N);

    size_t avl, vl;
    for (int i = 0, avl = GOLDEN_N; avl > 0; i += vl, avl -= vl) {
        vl = __riscv_vsetvl_e32m4(avl);
        vfloat32m4_t x = __riscv_vle32_v_f32m4(&input[i], vl);
        vfloat32m4_t e = __gloden_vec_exp(x, vl);
        vfloat32m4_t s = __gloden_vec_sin(x, vl);
        vfloat32m4_t c = __gloden_vec_cos(x, vl);
        __riscv_vse32_v_f32m4(&out_exp[i], e, vl);
        __riscv_vse32_v_f32m4(&out_sin[i], s, vl);
        __riscv_vse32_v_f32m4(&out_cos[i], c, vl);
    }

    printf("#ifndef GOLDEN_VEC_MATH_EXP_SIN_COS_N%d_H\n", GOLDEN_N);
    printf("#define GOLDEN_VEC_MATH_EXP_SIN_COS_N%d_H\n\n", GOLDEN_N);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_VEC_MATH_N %d\n\n", GOLDEN_N);

    print_f32_array("golden_vec_math_input_x", input, GOLDEN_N);
    printf("\n");
    print_f32_array("golden_vec_math_exp", out_exp, GOLDEN_N);
    printf("\n");
    print_f32_array("golden_vec_math_sin", out_sin, GOLDEN_N);
    printf("\n");
    print_f32_array("golden_vec_math_cos", out_cos, GOLDEN_N);

    printf("\n#endif /* GOLDEN_VEC_MATH_EXP_SIN_COS_N%d_H */\n", GOLDEN_N);
    return 0;
}
