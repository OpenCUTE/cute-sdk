#ifndef CUTE_QUANT_H
#define CUTE_QUANT_H

#include <assert.h>
#include <riscv_vector.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cute_vec_math.h"

static inline void cute_smoothquant_stage1_getscale(const float *input, float *scale,
                                                    int rows, int cols)
{
    assert(cols % (64 * 4) == 0);
    assert(rows % 16 == 0);
    assert(rows <= 1024);
    assert(cols <= 32768);

    for (int r = 0; r < rows; r++) {
        const float *row = &input[r * cols];
        size_t vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(cols);
        vl = vl_0;
        vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row[c], vl);
            vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
            tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
        }
        vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
        tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);
        float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);
        scale[r] = token_max / 127.0f;
    }
}

static inline void cute_smoothquant_stage2_quant(const float *input, int8_t *output,
                                                 const float *scale,
                                                 int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const float *row = &input[r * cols];
        int8_t *output_row = &output[r * cols];
        size_t vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(cols);
        vl = vl_0;
        float id = 1.0f / scale[r];
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row[c], vl);
            vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);
            vint16m2_t vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
            vint8m1_t vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);
            __riscv_vse8_v_i8m1(&output_row[c], vs, vl);
        }
    }
}

static inline void cute_smoothquant(float *input, int rows, int cols,
                                    int8_t *output, float *output_scale,
                                    bool need_stage1)
{
    if (need_stage1) {
        cute_smoothquant_stage1_getscale(input, output_scale, rows, cols);
    }
    cute_smoothquant_stage2_quant(input, output, output_scale, rows, cols);
}

static inline void cute_rmsnorm(const float *input, float *output,
                                const float *weight, float rms_epsilon,
                                int batch, int seq_len, int hidden_dim)
{
    assert(batch > 0 && seq_len > 0 && hidden_dim > 0);
    assert(hidden_dim % (16 * 4) == 0);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(
                    &input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m4_f32m1(sum_vec,
                    __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0f / cute_fast_sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(
                    &input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t weight_vec = __riscv_vle32_v_f32m4(&weight[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, weight_vec, vl);
                __riscv_vse32_v_f32m4(
                    &output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
            }
        }
    }
}

static inline void cute_rmsnorm_with_scale(const float *input, float *output,
                                           const float *weight, float *per_token_scale,
                                           float rms_epsilon,
                                           int batch, int seq_len, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(
                    &input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredusum_vs_f32m4_f32m1(sum_vec,
                    __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0f / cute_fast_sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(
                    &input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t weight_vec = __riscv_vle32_v_f32m4(&weight[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, weight_vec, vl);
                __riscv_vse32_v_f32m4(
                    &output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
                vfloat32m4_t abs_max_vec = __riscv_vfabs_v_f32m4(scaled_vec, vl);
                max_vec = __riscv_vfmax_vv_f32m4(max_vec, abs_max_vec, vl);
            }
            float token_max = __riscv_vfmv_f_s_f32m1_f32(
                __riscv_vfredmax_vs_f32m4_f32m1(max_vec,
                    __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            per_token_scale[b * seq_len + j] = token_max / 127.0f;
        }
    }
}

#endif /* CUTE_QUANT_H */
