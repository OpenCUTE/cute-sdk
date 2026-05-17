#ifndef CUTE_ELEMENTWISE_H
#define CUTE_ELEMENTWISE_H

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

#include "cute_vec_math.h"

static inline float cute_silu_scalar(float x)
{
    return x / (1.0f + expf(-x));
}

static inline void cute_silu_tile(float *data, uint64_t stride, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float *row = (float *)((char *)data + r * stride);
        size_t vl;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t x = __riscv_vle32_v_f32m4(&row[c], vl);
            vfloat32m4_t exp_neg_x = cute_vec_exp(__riscv_vfneg_v_f32m4(x, vl), vl);
            vfloat32m4_t silu = __riscv_vfdiv_vv_f32m4(
                x, __riscv_vfadd_vf_f32m4(exp_neg_x, 1.0f, vl), vl);
            __riscv_vse32_v_f32m4(&row[c], silu, vl);
        }
    }
}

static inline void cute_hadamard_tile(const float *lhs, uint64_t lhs_stride,
                                      const float *rhs, uint64_t rhs_stride,
                                      float *output, uint64_t output_stride,
                                      float *row_absmax,
                                      int rows, int cols)
{
    size_t vl_0 = __riscv_vsetvl_e32m4(cols);
    for (int r = 0; r < rows; r++) {
        const float *lhs_row = (const float *)((const char *)lhs + r * lhs_stride);
        const float *rhs_row = (const float *)((const char *)rhs + r * rhs_stride);
        float *out_row = (float *)((char *)output + r * output_stride);
        vfloat32m4_t absmax_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        size_t vl;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t l = __riscv_vle32_v_f32m4(&lhs_row[c], vl);
            vfloat32m4_t rr = __riscv_vle32_v_f32m4(&rhs_row[c], vl);
            vfloat32m4_t prod = __riscv_vfmul_vv_f32m4(l, rr, vl);
            __riscv_vse32_v_f32m4(&out_row[c], prod, vl);
            vfloat32m4_t abs_prod = __riscv_vfabs_v_f32m4(prod, vl);
            absmax_vec = __riscv_vfmax_vv_f32m4(absmax_vec, abs_prod, vl);
        }
        float token_max = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m4_f32m1(absmax_vec,
                __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
        if (token_max > row_absmax[r]) {
            row_absmax[r] = token_max;
        }
    }
}

static inline void cute_resadd_tile(const float *lhs, uint64_t lhs_stride,
                                    const float *rhs, uint64_t rhs_stride,
                                    float *output, uint64_t output_stride,
                                    int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const float *lhs_row = (const float *)((const char *)lhs + r * lhs_stride);
        const float *rhs_row = (const float *)((const char *)rhs + r * rhs_stride);
        float *out_row = (float *)((char *)output + r * output_stride);
        size_t vl;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t l = __riscv_vle32_v_f32m4(&lhs_row[c], vl);
            vfloat32m4_t rr = __riscv_vle32_v_f32m4(&rhs_row[c], vl);
            __riscv_vse32_v_f32m4(&out_row[c], __riscv_vfadd_vv_f32m4(l, rr, vl), vl);
        }
    }
}

#endif /* CUTE_ELEMENTWISE_H */
