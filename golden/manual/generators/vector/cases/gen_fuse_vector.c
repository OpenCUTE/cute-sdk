/*
 * Golden generator for vector fusion primitives.
 */

#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
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
#ifndef FUSE_KVSCALE
#define FUSE_KVSCALE 1.0f
#endif
#ifndef FUSE_ROPE_POS
#define FUSE_ROPE_POS 17
#endif

#define TOTAL (GOLDEN_M * GOLDEN_N)
#define HALF_N (GOLDEN_N / 2)

static void print_f32_array(const char *name, const float *data, int n)
{
    printf("static const float %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 4 == 0) {
            printf("    ");
        }
        printf("%a", data[i]);
        if (i < n - 1) {
            printf(",");
        }
        if (i % 4 == 3 || i == n - 1) {
            printf("\n");
        } else {
            printf(" ");
        }
    }
    printf("};\n");
}

static void print_i32_array(const char *name, const int32_t *data, int n)
{
    printf("static const int32_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 4 == 0) {
            printf("    ");
        }
        printf("%d", data[i]);
        if (i < n - 1) {
            printf(",");
        }
        if (i % 4 == 3 || i == n - 1) {
            printf("\n");
        } else {
            printf(" ");
        }
    }
    printf("};\n");
}

static void print_u16_array(const char *name, const uint16_t *data, int n)
{
    printf("static const uint16_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 8 == 0) {
            printf("    ");
        }
        printf("0x%04x", data[i]);
        if (i < n - 1) {
            printf(",");
        }
        if (i % 8 == 7 || i == n - 1) {
            printf("\n");
        } else {
            printf(" ");
        }
    }
    printf("};\n");
}

static void print_u8_array(const char *name, const uint8_t *data, int n)
{
    printf("static const uint8_t %s[%d] = {\n", name, n);
    for (int i = 0; i < n; i++) {
        if (i % 16 == 0) {
            printf("    ");
        }
        printf("0x%02x", data[i]);
        if (i < n - 1) {
            printf(",");
        }
        if (i % 16 == 15 || i == n - 1) {
            printf("\n");
        } else {
            printf(" ");
        }
    }
    printf("};\n");
}

static void scale_f32(const float *input, float *output, float scale, int count)
{
    size_t vl;
    for (int i = 0, avl = count; avl > 0; i += vl, avl -= vl) {
        vl = __riscv_vsetvl_e32m4(avl);
        vfloat32m4_t values = __riscv_vle32_v_f32m4(&input[i], vl);
        values = __riscv_vfmul_vf_f32m4(values, scale, vl);
        __riscv_vse32_v_f32m4(&output[i], values, vl);
    }
}

static void print_common_header(const char *guard)
{
    printf("#ifndef %s\n", guard);
    printf("#define %s\n\n", guard);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_FUSE_M %d\n", GOLDEN_M);
    printf("#define GOLDEN_FUSE_N %d\n", GOLDEN_N);
    printf("#define GOLDEN_FUSE_TOTAL %d\n", TOTAL);
    printf("#define GOLDEN_FUSE_KVSCALE %a\n", (float)FUSE_KVSCALE);
    printf("#define GOLDEN_FUSE_ROPE_POS %d\n\n", FUSE_ROPE_POS);
}

int main(void)
{
#if defined(FUSE_DEQUANT_SILU) || defined(FUSE_DEQUANT_RESADD) || defined(FUSE_DEQUANT_BF16CVT) || defined(FUSE_DEQUANT_ROPE_BF16CVT) || defined(FUSE_DEQUANT_HADAMARD)
    int32_t input_i32[TOTAL];
    float input_scale[GOLDEN_M];
    float weight_scale[1];
    float dequant[TOTAL];

    memset(input_i32, 0, sizeof(input_i32));
    memset(input_scale, 0, sizeof(input_scale));
    memset(weight_scale, 0, sizeof(weight_scale));
    memset(dequant, 0, sizeof(dequant));

    cute_fill_dequant_input_i32(input_i32, TOTAL);
    cute_fill_dequant_scale(input_scale, weight_scale, GOLDEN_M);
    __gloden_dequant_i32_to_f32(input_i32, dequant, input_scale, weight_scale, GOLDEN_M, GOLDEN_N);
#endif

#if defined(FUSE_DEQUANT_SILU)
    float output[TOTAL];
    memset(output, 0, sizeof(output));
    __gloden_silu(dequant, output, 1, GOLDEN_M, GOLDEN_N);

    print_common_header("GOLDEN_FUSE_DEQUANT_SILU_H");
    print_i32_array("golden_fuse_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_fuse_weight_scale", weight_scale, 1);
    printf("\n");
    print_f32_array("golden_fuse_output", output, TOTAL);
    printf("\n#endif /* GOLDEN_FUSE_DEQUANT_SILU_H */\n");

#elif defined(FUSE_DEQUANT_RESADD)
    float residual[TOTAL];
    float unused[TOTAL];
    float output[TOTAL];
    memset(residual, 0, sizeof(residual));
    memset(unused, 0, sizeof(unused));
    memset(output, 0, sizeof(output));
    cute_fill_resadd_input(unused, residual, TOTAL);
    __gloden_resadd_f32(dequant, residual, output, GOLDEN_M, GOLDEN_N);

    print_common_header("GOLDEN_FUSE_DEQUANT_RESADD_H");
    print_i32_array("golden_fuse_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_fuse_weight_scale", weight_scale, 1);
    printf("\n");
    print_f32_array("golden_fuse_residual", residual, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_output", output, TOTAL);
    printf("\n#endif /* GOLDEN_FUSE_DEQUANT_RESADD_H */\n");

#elif defined(FUSE_DEQUANT_BF16CVT)
    uint16_t output_f16[TOTAL];
    memset(output_f16, 0, sizeof(output_f16));
    __gloden_cvrtbf16(dequant, output_f16, GOLDEN_M, GOLDEN_N);

    print_common_header("GOLDEN_FUSE_DEQUANT_BF16CVT_H");
    print_i32_array("golden_fuse_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_fuse_weight_scale", weight_scale, 1);
    printf("\n");
    print_u16_array("golden_fuse_output_f16", output_f16, TOTAL);
    printf("\n#endif /* GOLDEN_FUSE_DEQUANT_BF16CVT_H */\n");

#elif defined(FUSE_DEQUANT_ROPE_BF16CVT)
    float rope_theta[HALF_N];
    float rope_f32[TOTAL];
    uint16_t output_f16[TOTAL];
    memset(rope_theta, 0, sizeof(rope_theta));
    memset(rope_f32, 0, sizeof(rope_f32));
    memset(output_f16, 0, sizeof(output_f16));
    cute_fill_rope_theta(rope_theta, HALF_N);
    __gloden_rope(dequant, rope_f32, rope_theta, FUSE_ROPE_POS, 1, 1, GOLDEN_M, GOLDEN_N);
    __gloden_cvrtbf16(rope_f32, output_f16, GOLDEN_M, GOLDEN_N);

    print_common_header("GOLDEN_FUSE_DEQUANT_ROPE_BF16CVT_H");
    print_i32_array("golden_fuse_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_fuse_weight_scale", weight_scale, 1);
    printf("\n");
    print_f32_array("golden_fuse_rope_theta", rope_theta, HALF_N);
    printf("\n");
    print_u16_array("golden_fuse_output_f16", output_f16, TOTAL);
    printf("\n#endif /* GOLDEN_FUSE_DEQUANT_ROPE_BF16CVT_H */\n");

#elif defined(FUSE_DEQUANT_HADAMARD)
    float lhs[TOTAL];
    float unused[TOTAL];
    float output[TOTAL];
    float row_absmax[GOLDEN_M];
    memset(lhs, 0, sizeof(lhs));
    memset(unused, 0, sizeof(unused));
    memset(output, 0, sizeof(output));
    memset(row_absmax, 0, sizeof(row_absmax));
    cute_fill_hadamard_input(lhs, unused, TOTAL);
    __gloden_hadamard_f32(lhs, dequant, output, row_absmax, GOLDEN_M, GOLDEN_N);

    print_common_header("GOLDEN_FUSE_DEQUANT_HADAMARD_H");
    print_i32_array("golden_fuse_input_i32", input_i32, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_input_scale", input_scale, GOLDEN_M);
    printf("\n");
    print_f32_array("golden_fuse_weight_scale", weight_scale, 1);
    printf("\n");
    print_f32_array("golden_fuse_lhs", lhs, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_output", output, TOTAL);
    printf("\n");
    print_f32_array("golden_fuse_row_absmax", row_absmax, GOLDEN_M);
    printf("\n#endif /* GOLDEN_FUSE_DEQUANT_HADAMARD_H */\n");

#elif defined(FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVT)
    float input[TOTAL];
    float scaled[TOTAL];
    float softmax[TOTAL];
    uint8_t mask[(GOLDEN_M * GOLDEN_N + 7) / 8];
    uint16_t output_f16[TOTAL];
    memset(input, 0, sizeof(input));
    memset(scaled, 0, sizeof(scaled));
    memset(softmax, 0, sizeof(softmax));
    memset(mask, 0, sizeof(mask));
    memset(output_f16, 0, sizeof(output_f16));
    cute_fill_softmax_input(input, TOTAL);
    cute_fill_causal_mask(mask, GOLDEN_M, GOLDEN_N);
    scale_f32(input, scaled, (float)FUSE_KVSCALE, TOTAL);
    __gloden_softmax(scaled, softmax, mask, GOLDEN_M, GOLDEN_N);
    __gloden_cvrtbf16(softmax, output_f16, GOLDEN_M, GOLDEN_N);

    print_common_header("GOLDEN_FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVT_H");
    print_f32_array("golden_fuse_input", input, TOTAL);
    printf("\n");
    print_u8_array("golden_fuse_causal_mask", mask, (GOLDEN_M * GOLDEN_N + 7) / 8);
    printf("\n");
    print_u16_array("golden_fuse_output_f16", output_f16, TOTAL);
    printf("\n#endif /* GOLDEN_FUSE_MASKED_SOFTMAX_KVSCALE_BF16CVT_H */\n");
#else
#error "Select one FUSE_* generator mode"
#endif

    return 0;
}
