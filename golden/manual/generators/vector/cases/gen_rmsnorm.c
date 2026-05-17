/*
 * Golden generator for RMSNorm (plain and with_scale).
 * Params: GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM, RMSNORM_PLAIN or RMSNORM_WITH_SCALE
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "cute_golden_inputs.h"
#include "nvwa_gloden_opt.h"
#include "nvwa_llama_primitives.h"

#ifndef GOLDEN_BATCH
#define GOLDEN_BATCH 1
#endif
#ifndef GOLDEN_SEQ_LEN
#define GOLDEN_SEQ_LEN 128
#endif
#ifndef GOLDEN_HIDDEN_DIM
#define GOLDEN_HIDDEN_DIM 2048
#endif
#define TOTAL (GOLDEN_BATCH * GOLDEN_SEQ_LEN * GOLDEN_HIDDEN_DIM)

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
    float weight[GOLDEN_HIDDEN_DIM];
    float per_token_scale[GOLDEN_BATCH * GOLDEN_SEQ_LEN];
    float eps = 1e-5f;

    memset(input, 0, sizeof(input));
    memset(output, 0, sizeof(output));
    memset(weight, 0, sizeof(weight));
    memset(per_token_scale, 0, sizeof(per_token_scale));

    cute_fill_rmsnorm_input(input, TOTAL);
    cute_fill_rmsnorm_weight(weight, GOLDEN_HIDDEN_DIM);

#ifdef RMSNORM_WITH_SCALE
    __gloden_RMSnorm_with_getabsmax_scale(input, output, weight, per_token_scale,
                                           eps, GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);

    printf("#ifndef GOLDEN_RMSNORM_SCALE_B%d_S%d_H%d_H\n", GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);
    printf("#define GOLDEN_RMSNORM_SCALE_B%d_S%d_H%d_H\n\n", GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_RMSNORM_BATCH %d\n", GOLDEN_BATCH);
    printf("#define GOLDEN_RMSNORM_SEQ_LEN %d\n", GOLDEN_SEQ_LEN);
    printf("#define GOLDEN_RMSNORM_HIDDEN_DIM %d\n\n", GOLDEN_HIDDEN_DIM);

    print_f32_array("golden_rmsnorm_input", input, TOTAL);
    printf("\n");
    print_f32_array("golden_rmsnorm_weight", weight, GOLDEN_HIDDEN_DIM);
    printf("\n");
    print_f32_array("golden_rmsnorm_output", output, TOTAL);
    printf("\n");
    print_f32_array("golden_rmsnorm_per_token_scale", per_token_scale, GOLDEN_BATCH * GOLDEN_SEQ_LEN);

    printf("\n#endif /* GOLDEN_RMSNORM_SCALE_B%d_S%d_H%d_H */\n", GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);

#else
    __gloden_RMSnorm(input, output, weight, eps, GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);

    printf("#ifndef GOLDEN_RMSNORM_B%d_S%d_H%d_H\n", GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);
    printf("#define GOLDEN_RMSNORM_B%d_S%d_H%d_H\n\n", GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);
    printf("#include <stdint.h>\n\n");
    printf("#define GOLDEN_RMSNORM_BATCH %d\n", GOLDEN_BATCH);
    printf("#define GOLDEN_RMSNORM_SEQ_LEN %d\n", GOLDEN_SEQ_LEN);
    printf("#define GOLDEN_RMSNORM_HIDDEN_DIM %d\n\n", GOLDEN_HIDDEN_DIM);

    print_f32_array("golden_rmsnorm_input", input, TOTAL);
    printf("\n");
    print_f32_array("golden_rmsnorm_weight", weight, GOLDEN_HIDDEN_DIM);
    printf("\n");
    print_f32_array("golden_rmsnorm_output", output, TOTAL);

    printf("\n#endif /* GOLDEN_RMSNORM_B%d_S%d_H%d_H */\n", GOLDEN_BATCH, GOLDEN_SEQ_LEN, GOLDEN_HIDDEN_DIM);
#endif

    return 0;
}
