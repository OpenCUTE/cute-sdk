/*
 * Primitives split from NVWA llama3_1B_cpu.c fusion functions.
 * dequant, resadd, hadamard, RMSnorm_with_getabsmax_scale.
 *
 * Splitting principle: remove fusion glue, keep constants/intrinsic order/rounding unchanged.
 */

#ifndef NVWA_LLAMA_PRIMITIVES_H
#define NVWA_LLAMA_PRIMITIVES_H

#include <riscv_vector.h>
#include <stdint.h>
#include <math.h>
#include "nvwa_gloden_opt.h" /* fast_sqrt */

/* ------------------------------------------------------------------ */
/*  dequant I32 -> F32                                                */
/*  from fuse_ops_DEQUANT_RESADD stage 1                              */
/* ------------------------------------------------------------------ */
static inline void __gloden_dequant_i32_to_f32(int32_t* input, float* output,
                                                float* input_scale, float* weight_scale,
                                                int M, int N)
{
    for (int s = 0; s < M; s++) {
        float scale = input_scale[s] * weight_scale[0];
        float* output_row = output + s * N;
        int32_t* input_row = input + s * N;
        size_t avl, vl;
        for (int i = 0, avl = N; avl > 0; i += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(&input_row[i], vl);
            vfloat32m4_t deq = __riscv_vfmul_vf_f32m4(
                __riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);
            __riscv_vse32_v_f32m4(&output_row[i], deq, vl);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  resadd (element-wise F32 add)                                     */
/*  from fuse_ops_DEQUANT_RESADD stage 2                              */
/* ------------------------------------------------------------------ */
static inline void __gloden_resadd_f32(float* lhs, float* rhs, float* output, int M, int N)
{
    for (int s = 0; s < M; s++) {
        float* lhs_row = lhs + s * N;
        float* rhs_row = rhs + s * N;
        float* out_row = output + s * N;
        size_t avl, vl;
        for (int i = 0, avl = N; avl > 0; i += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t l = __riscv_vle32_v_f32m4(&lhs_row[i], vl);
            vfloat32m4_t r = __riscv_vle32_v_f32m4(&rhs_row[i], vl);
            __riscv_vse32_v_f32m4(&out_row[i], __riscv_vfadd_vv_f32m4(l, r, vl), vl);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  hadamard (element-wise F32 multiply) + row_absmax                 */
/*  from fuse_ops_DEQUANT_HADAMARD stage 2+3                          */
/*  row_absmax computation restored from commented-out NVWA code       */
/* ------------------------------------------------------------------ */
static inline void __gloden_hadamard_f32(float* lhs, float* rhs, float* output,
                                          float* row_absmax, int M, int N)
{
    size_t vl_0 = __riscv_vsetvl_e32m4(N);
    for (int s = 0; s < M; s++) {
        float* lhs_row = lhs + s * N;
        float* rhs_row = rhs + s * N;
        float* out_row = output + s * N;
        vfloat32m4_t absmax_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        size_t avl, vl;
        for (int i = 0, avl = N; avl > 0; i += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t l = __riscv_vle32_v_f32m4(&lhs_row[i], vl);
            vfloat32m4_t r = __riscv_vle32_v_f32m4(&rhs_row[i], vl);
            vfloat32m4_t prod = __riscv_vfmul_vv_f32m4(l, r, vl);
            __riscv_vse32_v_f32m4(&out_row[i], prod, vl);
            vfloat32m4_t abs_prod = __riscv_vfabs_v_f32m4(prod, vl);
            absmax_vec = __riscv_vfmax_vv_f32m4(absmax_vec, abs_prod, vl);
        }
        float token_max = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m4_f32m1(absmax_vec,
                __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
        row_absmax[s] = token_max;
    }
}

/* ------------------------------------------------------------------ */
/*  RMSnorm with getabsmax_scale                                      */
/*  from llama3_1B_cpu.c:1067 RMSnorm_With_getabsmax_scale           */
/* ------------------------------------------------------------------ */
static inline void __gloden_RMSnorm_with_getabsmax_scale(
    float* input, float* output, float* per_channel_scale,
    float* per_token_scale, float rms_epsilon,
    int batch, int seq_len, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
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
            float rms = 1.0f / fast_sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(
                    &input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channel_scale_vec = __riscv_vle32_v_f32m4(&per_channel_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channel_scale_vec, vl);
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

#endif /* NVWA_LLAMA_PRIMITIVES_H */
