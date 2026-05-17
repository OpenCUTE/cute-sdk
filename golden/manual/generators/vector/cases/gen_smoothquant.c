/*
 * Golden generator for SmoothQuant: F32 -> I8 with per-row scale.
 * Params: GOLDEN_M, GOLDEN_K
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_gloden_opt.h"

#ifndef GOLDEN_M
#define GOLDEN_M 128
#endif
#ifndef GOLDEN_K
#define GOLDEN_K 2048
#endif
#define TOTAL (GOLDEN_M * GOLDEN_K)

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

static void print_i8_array(const char *name, const int8_t *data, int n) {
    printf("static const int8_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 16 == 0) printf("    ");
        printf("%d", (int)data[i]);
        if (i < n - 1) printf(", ");
        if (i % 16 == 15 || i == n - 1) printf("\n");
    }
    printf("};\n");
}

int main(void) {
    float input[TOTAL];
    int8_t output_i8[TOTAL];
    float scale[GOLDEN_M];

    memset(input, 0, sizeof(input));
    memset(output_i8, 0, sizeof(output_i8));
    memset(scale, 0, sizeof(scale));

    cute_fill_smoothquant_input(input, TOTAL);
    __gloden_smoothquantO1(input, output_i8, scale, GOLDEN_M, GOLDEN_K);

    printf("#ifndef GOLDEN_SMOOTHQUANT_M%d_K%d_H\n", GOLDEN_M, GOLDEN_K);
    printf("#define GOLDEN_SMOOTHQUANT_M%d_K%d_H\n\n", GOLDEN_M, GOLDEN_K);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_SMOOTHQUANT_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_SMOOTHQUANT_K %d\n\n", GOLDEN_K);

    print_f32_array("golden_smoothquant_input", input, TOTAL);
    printf("\n");
    print_i8_array("golden_smoothquant_output_i8", output_i8, TOTAL);
    printf("\n");
    print_f32_array("golden_smoothquant_scale", scale, GOLDEN_M);

    printf("\n#endif /* GOLDEN_SMOOTHQUANT_M%d_K%d_H */\n", GOLDEN_M, GOLDEN_K);
    return 0;
}
