/*
 * Golden generator for RoPE with BF16 output.
 * Params: GOLDEN_M, GOLDEN_HEAD_DIM, GOLDEN_N_HEAD, ROPE_POS
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_gloden_opt.h"

#ifndef GOLDEN_M
#define GOLDEN_M 64
#endif
#ifndef GOLDEN_HEAD_DIM
#define GOLDEN_HEAD_DIM 64
#endif
#ifndef GOLDEN_N_HEAD
#define GOLDEN_N_HEAD 1
#endif
#ifndef ROPE_POS
#define ROPE_POS 0
#endif
#define TOTAL (GOLDEN_M * GOLDEN_HEAD_DIM)
#define HALF_DIM (GOLDEN_HEAD_DIM / 2)

static void print_f32_array(const char *name, const float *data, int n) {
    printf("static const float %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 4 == 0) printf("    ");
        printf("%a", data[i]);
        if (i < n - 1) printf(",");
        if (i % 4 == 3 || i == n - 1) printf("\n");
        else printf(" ");
    }
    printf("};\n");
}

static void print_u16_array(const char *name, const uint16_t *data, int n) {
    printf("static const uint16_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 8 == 0) printf("    ");
        printf("0x%04x", data[i]);
        if (i < n - 1) printf(",");
        if (i % 8 == 7 || i == n - 1) printf("\n");
        else printf(" ");
    }
    printf("};\n");
}

int main(void) {
    float input[TOTAL];
    float output_f32[TOTAL];
    float rope_theta[HALF_DIM];
    uint16_t output_f16[TOTAL];

    memset(input, 0, sizeof(input));
    memset(output_f32, 0, sizeof(output_f32));
    memset(output_f16, 0, sizeof(output_f16));

    cute_fill_rope_input(input, TOTAL);
    cute_fill_rope_theta(rope_theta, HALF_DIM);

    __gloden_rope(input, output_f32, rope_theta, ROPE_POS, 1, GOLDEN_N_HEAD, GOLDEN_M, GOLDEN_HEAD_DIM);
    __gloden_cvrtbf16(output_f32, output_f16, GOLDEN_M, GOLDEN_HEAD_DIM);

    printf("#ifndef GOLDEN_ROPE_M%d_HEAD%d_POS%d_F16_H\n", GOLDEN_M, GOLDEN_HEAD_DIM, ROPE_POS);
    printf("#define GOLDEN_ROPE_M%d_HEAD%d_POS%d_F16_H\n\n", GOLDEN_M, GOLDEN_HEAD_DIM, ROPE_POS);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_ROPE_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_ROPE_HEAD_DIM %d\n", GOLDEN_HEAD_DIM);
    printf("#define GOLDEN_ROPE_POS %d\n\n", ROPE_POS);

    print_f32_array("golden_rope_input", input, TOTAL);
    printf("\n");
    print_f32_array("golden_rope_theta", rope_theta, HALF_DIM);
    printf("\n");
    print_u16_array("golden_rope_output_f16", output_f16, TOTAL);

    printf("\n#endif /* GOLDEN_ROPE_M%d_HEAD%d_POS%d_F16_H */\n", GOLDEN_M, GOLDEN_HEAD_DIM, ROPE_POS);
    return 0;
}
