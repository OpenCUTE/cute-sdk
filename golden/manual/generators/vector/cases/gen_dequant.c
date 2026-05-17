/*
 * Golden generator for dequant I32 -> F32 or F16.
 * Params: GOLDEN_M, GOLDEN_N, DEQUANT_OUTPUT_F32 or DEQUANT_OUTPUT_F16
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_gloden_opt.h"
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

static void print_u16_array(const char *name, const uint16_t *data, int n) {
    printf("static const uint16_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 8 == 0) printf("    ");
        printf("0x%04x", data[i]);
        if (i < n - 1) printf(", ");
        if (i % 8 == 7 || i == n - 1) printf("\n");
    }
    printf("};\n");
}

static void print_i32_array(const char *name, const int32_t *data, int n) {
    printf("static const int32_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 4 == 0) printf("    ");
        printf("%d", data[i]);
        if (i < n - 1) printf(", ");
        if (i % 4 == 3 || i == n - 1) printf("\n");
    }
    printf("};\n");
}

int main(void) {
    int32_t input_i32[TOTAL];
    float input_scale[GOLDEN_M];
    float weight_scale[1];
    float output_f32[TOTAL];

    memset(output_f32, 0, sizeof(output_f32));

    cute_fill_dequant_input_i32(input_i32, TOTAL);
    cute_fill_dequant_scale(input_scale, weight_scale, GOLDEN_M);
    __gloden_dequant_i32_to_f32(input_i32, output_f32, input_scale, weight_scale, GOLDEN_M, GOLDEN_N);

#if defined(DEQUANT_OUTPUT_F16)
    uint16_t output_f16[TOTAL];
    memset(output_f16, 0, sizeof(output_f16));
    __gloden_cvrtfp16(output_f32, output_f16, GOLDEN_M, GOLDEN_N);

    printf("#ifndef GOLDEN_DEQUANT_M%d_N%d_I32_TO_F16_H\n", GOLDEN_M, GOLDEN_N);
    printf("#define GOLDEN_DEQUANT_M%d_N%d_I32_TO_F16_H\n\n", GOLDEN_M, GOLDEN_N);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_DEQUANT_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_DEQUANT_N %d\n\n", GOLDEN_N);

    print_i32_array("golden_dequant_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_dequant_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_dequant_weight_scale", weight_scale, 1);
    printf("\n");
    print_u16_array("golden_dequant_output_f16", output_f16, TOTAL);

    printf("\n#endif /* GOLDEN_DEQUANT_M%d_N%d_I32_TO_F16_H */\n", GOLDEN_M, GOLDEN_N);

#else
    printf("#ifndef GOLDEN_DEQUANT_M%d_N%d_I32_TO_F32_H\n", GOLDEN_M, GOLDEN_N);
    printf("#define GOLDEN_DEQUANT_M%d_N%d_I32_TO_F32_H\n\n", GOLDEN_M, GOLDEN_N);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_DEQUANT_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_DEQUANT_N %d\n\n", GOLDEN_N);

    print_i32_array("golden_dequant_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_dequant_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_dequant_weight_scale", weight_scale, 1);
    printf("\n");
    print_f32_array("golden_dequant_output_f32", output_f32, TOTAL);

    printf("\n#endif /* GOLDEN_DEQUANT_M%d_N%d_I32_TO_F32_H */\n", GOLDEN_M, GOLDEN_N);
#endif

    return 0;
}
