#ifndef CUTE_SEQUENCE_H
#define CUTE_SEQUENCE_H

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "cute_vec_math.h"

static inline void cute_rope_f16_tile(const float *input, uint64_t input_stride,
                                      void *output, uint64_t output_stride,
                                      const float *rope_theta,
                                      int pos, int key_dim,
                                      int rows, int cols)
{
    (void)cols;
    const int half_dim = key_dim / 2;
    for (int r = 0; r < rows; r++) {
        int pos_r = r + pos;
        const float *input_row = (const float *)((const char *)input + r * input_stride);
        _Float16 *output_row = (_Float16 *)((char *)output + r * output_stride);
        size_t vl;
        for (int k = 0, avl = half_dim; avl > 0; k += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t theta_vec = __riscv_vle32_v_f32m4(rope_theta + k, vl);
            vfloat32m4_t angle_vec = __riscv_vfmul_vf_f32m4(theta_vec, pos_r, vl);

            vfloat32m4_t sin_vec = cute_vec_sin(angle_vec, vl);
            vfloat32m4_t cos_vec = cute_vec_cos(angle_vec, vl);

            vfloat32m4_t real_in = __riscv_vlse32_v_f32m4(input_row + 2 * k, 2 * sizeof(float), vl);
            vfloat32m4_t imag_in = __riscv_vlse32_v_f32m4(input_row + 2 * k + 1, 2 * sizeof(float), vl);

            vfloat32m4_t real_out = __riscv_vfmsub_vv_f32m4(real_in, cos_vec,
                __riscv_vfmul_vv_f32m4(imag_in, sin_vec, vl), vl);
            vfloat32m4_t imag_out = __riscv_vfmacc_vv_f32m4(
                __riscv_vfmul_vv_f32m4(real_in, sin_vec, vl), imag_in, cos_vec, vl);

            vfloat16m2_t real_f16 = __riscv_vfncvt_f_f_w_f16m2(real_out, vl);
            vfloat16m2_t imag_f16 = __riscv_vfncvt_f_f_w_f16m2(imag_out, vl);
            __riscv_vse16_v_f16m2(output_row + k, real_f16, vl);
            __riscv_vse16_v_f16m2(output_row + half_dim + k, imag_f16, vl);
        }
    }
}

static inline void cute_rope_bf16_tile(const float *input, uint64_t input_stride,
                                       void *output, uint64_t output_stride,
                                       const float *rope_theta,
                                       int pos, int key_dim,
                                       int rows, int cols)
{
    cute_rope_f16_tile(input, input_stride, output, output_stride,
                       rope_theta, pos, key_dim, rows, cols);
}

static inline void cute_masked_softmax_f16_tile(const float *input, uint64_t input_stride,
                                                void *output, uint64_t output_stride,
                                                const int8_t *mask, uint64_t mask_stride,
                                                float scale,
                                                int row0, int col0,
                                                int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const float *input_row = (const float *)((const char *)input + r * input_stride);
        _Float16 *output_row = (_Float16 *)((char *)output + r * output_stride);
        const uint8_t *mask_row = (const uint8_t *)mask + (uint64_t)(row0 + r) * mask_stride;
        float scratch[cols];
        size_t vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(cols);

        float max_val = input_row[0] * scale;
        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(max_val, vl_0);
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row[c], vl);
            vec = __riscv_vfmul_vf_f32m4(vec, scale, vl);
            vbool8_t mask_vec = __riscv_vlm_v_b8((uint8_t *)(mask_row + (col0 + c) / 8), vl);
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), vec, mask_vec, vl);
            max_vec = __riscv_vfmax_vv_f32m4(max_vec, vec, vl);
        }
        max_val = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m4_f32m1(max_vec,
                __riscv_vfmv_v_f_f32m1(-INFINITY, vl_0), vl_0));

        float sum_exp = 0.0f;
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row[c], vl);
            vec = __riscv_vfmul_vf_f32m4(vec, scale, vl);
            vec = __riscv_vfsub_vf_f32m4(vec, max_val, vl);
            vbool8_t mask_vec = __riscv_vlm_v_b8((uint8_t *)(mask_row + (col0 + c) / 8), vl);
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-90, vl), vec, mask_vec, vl);
            vfloat32m4_t exp_vec = cute_vec_exp(vec, vl);
            __riscv_vse32_v_f32m4(&scratch[c], exp_vec, vl);
            sumexp_vec = __riscv_vfadd_vv_f32m4(sumexp_vec, exp_vec, vl);
        }
        sum_exp = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec,
                __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));

        vfloat32m4_t inv_sum_exp_vec = __riscv_vfmv_v_f_f32m4(1.0f / sum_exp, vl_0);
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&scratch[c], vl);
            vfloat32m4_t normalized = __riscv_vfmul_vv_f32m4(vec, inv_sum_exp_vec, vl);
            vfloat16m2_t normalized_f16 = __riscv_vfncvt_f_f_w_f16m2(normalized, vl);
            __riscv_vse16_v_f16m2(&output_row[c], normalized_f16, vl);
        }
    }
}

static inline void cute_masked_softmax_bf16_tile(const float *input, uint64_t input_stride,
                                                 void *output, uint64_t output_stride,
                                                 const int8_t *mask, uint64_t mask_stride,
                                                 float scale,
                                                 int row0, int col0,
                                                 int rows, int cols)
{
    cute_masked_softmax_f16_tile(input, input_stride, output, output_stride,
                                 mask, mask_stride, scale, row0, col0, rows, cols);
}

#endif /* CUTE_SEQUENCE_H */
