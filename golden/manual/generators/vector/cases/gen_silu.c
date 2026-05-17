/*
 * Golden generator for SiLU: y = x * sigmoid(x)
 * Params: GOLDEN_M, GOLDEN_N
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_gloden_opt.h"

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
    float input[TOTAL];
    float output[TOTAL];

    memset(input, 0, sizeof(input));
    memset(output, 0, sizeof(output));

    cute_fill_silu_input(input, TOTAL);
    __gloden_silu(input, output, 1, GOLDEN_M, GOLDEN_N);

    printf("#ifndef GOLDEN_SILU_M%d_N%d_F32_H\n", GOLDEN_M, GOLDEN_N);
    printf("#define GOLDEN_SILU_M%d_N%d_F32_H\n\n", GOLDEN_M, GOLDEN_N);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_SILU_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_SILU_N %d\n", GOLDEN_N);
    printf("#define GOLDEN_SILU_TOTAL %d\n\n", TOTAL);

    print_f32_array("golden_silu_input_x", input, TOTAL);
    printf("\n");
    print_f32_array("golden_silu_golden_y", output, TOTAL);

    printf("\n#endif /* GOLDEN_SILU_M%d_N%d_F32_H */\n", GOLDEN_M, GOLDEN_N);
    return 0;
}
