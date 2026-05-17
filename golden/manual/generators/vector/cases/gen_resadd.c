/*
 * Golden generator for resadd: Y = A + B (element-wise F32).
 * Params: GOLDEN_M, GOLDEN_N
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_llama_primitives.h"

#ifndef GOLDEN_M
#define GOLDEN_M 64
#endif
#ifndef GOLDEN_N
#define GOLDEN_N 64
#endif
#define TOTAL (GOLDEN_M * GOLDEN_N)

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
    float lhs[TOTAL], rhs[TOTAL], output[TOTAL];

    memset(lhs, 0, sizeof(lhs));
    memset(rhs, 0, sizeof(rhs));
    memset(output, 0, sizeof(output));

    cute_fill_resadd_input(lhs, rhs, TOTAL);
    __gloden_resadd_f32(lhs, rhs, output, GOLDEN_M, GOLDEN_N);

    printf("#ifndef GOLDEN_RESADD_M%d_N%d_F32_H\n", GOLDEN_M, GOLDEN_N);
    printf("#define GOLDEN_RESADD_M%d_N%d_F32_H\n\n", GOLDEN_M, GOLDEN_N);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_RESADD_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_RESADD_N %d\n\n", GOLDEN_N);

    print_f32_array("golden_resadd_lhs", lhs, TOTAL);
    printf("\n");
    print_f32_array("golden_resadd_rhs", rhs, TOTAL);
    printf("\n");
    print_f32_array("golden_resadd_output", output, TOTAL);

    printf("\n#endif /* GOLDEN_RESADD_M%d_N%d_F32_H */\n", GOLDEN_M, GOLDEN_N);
    return 0;
}
