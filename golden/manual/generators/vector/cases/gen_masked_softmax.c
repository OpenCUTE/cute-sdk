/*
 * Golden generator for masked softmax with causal mask, BF16 output.
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
#define GOLDEN_N 128
#endif
#define TOTAL (GOLDEN_M * GOLDEN_N)

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

static void print_u8_array(const char *name, const uint8_t *data, int n) {
    printf("static const uint8_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 16 == 0) printf("    ");
        printf("0x%02x", data[i]);
        if (i < n - 1) printf(",");
        if (i % 16 == 15 || i == n - 1) printf("\n");
        else printf(" ");
    }
    printf("};\n");
}

int main(void) {
    float input[TOTAL], output_f32[TOTAL];
    uint8_t mask[(GOLDEN_M * GOLDEN_N + 7) / 8];
    uint16_t output_bf16[TOTAL];

    memset(input, 0, sizeof(input));
    memset(output_f32, 0, sizeof(output_f32));
    memset(mask, 0, sizeof(mask));
    memset(output_bf16, 0, sizeof(output_bf16));

    cute_fill_softmax_input(input, TOTAL);
    cute_fill_causal_mask(mask, GOLDEN_M, GOLDEN_N);

    __gloden_softmax(input, output_f32, mask, GOLDEN_M, GOLDEN_N);
    __gloden_cvrtbf16(output_f32, output_bf16, GOLDEN_M, GOLDEN_N);

    int mask_bytes = (GOLDEN_M * GOLDEN_N + 7) / 8;

    printf("#ifndef GOLDEN_MASKED_SOFTMAX_M%d_N%d_CAUSAL_BF16_H\n", GOLDEN_M, GOLDEN_N);
    printf("#define GOLDEN_MASKED_SOFTMAX_M%d_N%d_CAUSAL_BF16_H\n\n", GOLDEN_M, GOLDEN_N);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_SOFTMAX_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_SOFTMAX_N %d\n\n", GOLDEN_N);

    print_f32_array("golden_softmax_input", input, TOTAL);
    printf("\n");
    print_u8_array("golden_softmax_causal_mask", mask, mask_bytes);
    printf("\n");
    print_u16_array("golden_softmax_output_bf16", output_bf16, TOTAL);

    printf("\n#endif /* GOLDEN_MASKED_SOFTMAX_M%d_N%d_CAUSAL_BF16_H */\n", GOLDEN_M, GOLDEN_N);
    return 0;
}
