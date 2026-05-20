/*
 * Cleaned copy of NVWA gloden_opt.h for Phase C0 golden generation.
 * Original: NVWA/llama3.2_1B/data_flow/gloden_opt.h
 * "gloden" is the original NVWA spelling, kept for consistency.
 *
 * Removed: f16_matmul, GeLu, Q_matmul_I8I8I32, pertoken_pertensor_scale, check_diff_*
 */

#ifndef NVWA_GLODEN_OPT_H
#define NVWA_GLODEN_OPT_H

#include <riscv_vector.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

static inline float fast_sqrt(float x) {
    float result;
    __asm__ volatile ("fsqrt.s %0, %1" : "=f"(result) : "f"(x));
    return result;
}

/* ------------------------------------------------------------------ */
/*  vec_exp                                                            */
/* ------------------------------------------------------------------ */
static inline vfloat32m4_t __gloden_vec_exp(vfloat32m4_t x, size_t vl) {
    const float NEG_LN2 = -0.69314718056f;
    const float INV_LN2 = 1.44269504089f;
    const int32_t MAX_A = 127;
    const int32_t MIN_A = -126;

    vfloat32m4_t af = __riscv_vfmul_vf_f32m4(x, INV_LN2, vl);
    vfloat32m4_t r = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl);
    vfloat32m4_t a = __riscv_vfadd_vv_f32m4(af, r, vl);
    a = __riscv_vfsub_vv_f32m4(a, r, vl);
    vint32m4_t a_int = __riscv_vfcvt_x_f_v_i32m4(a, vl);

    vbool8_t mask_max = __riscv_vmsgt_vx_i32m4_b8(a_int, MAX_A, vl);
    vbool8_t mask_min = __riscv_vmslt_vx_i32m4_b8(a_int, MIN_A, vl);

    vint32m4_t biased_exponent = __riscv_vadd_vx_i32m4(a_int, 127, vl);
    biased_exponent = __riscv_vsll_vx_i32m4(biased_exponent, 23, vl);
    vfloat32m4_t a2 = __riscv_vreinterpret_v_i32m4_f32m4(biased_exponent);

    vfloat32m4_t b = __riscv_vfmacc_vf_f32m4(x, NEG_LN2, a, vl);

    vfloat32m4_t c0 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c1 = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c2 = __riscv_vfmv_v_f_f32m4(0.5f, vl);
    vfloat32m4_t c3 = __riscv_vfmv_v_f_f32m4(0.166666666667f, vl);
    vfloat32m4_t c4 = __riscv_vfmv_v_f_f32m4(0.041666666667f, vl);
    vfloat32m4_t c5 = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c6 = __riscv_vfmv_v_f_f32m4(0.001388888889f, vl);

    vfloat32m4_t p;
    p = __riscv_vfmacc_vv_f32m4(c5, c6, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c4,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c3,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c2,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c1,  p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c0,  p, b, vl);
    p = __riscv_vfmul_vv_f32m4(a2, p, vl);

    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(INFINITY, vl), mask_max, vl);
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(0.0f, vl), mask_min, vl);
    return p;
}

/* ------------------------------------------------------------------ */
/*  vec_tanh                                                          */
/* ------------------------------------------------------------------ */
static inline vfloat32m4_t __gloden_vec_tanh(vfloat32m4_t x, size_t vl) {
    const float THRESHOLD = 12.0f;
    vfloat32m4_t abs_x = __riscv_vfabs_v_f32m4(x, vl);
    vfloat32m4_t result = __riscv_vfmv_v_f_f32m4(0.0f, vl);
    vbool8_t mask_large = __riscv_vmfgt_vf_f32m4_b8(abs_x, THRESHOLD, vl);
    vfloat32m4_t sign_x = __riscv_vfsgnj_vv_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), x, vl);
    result = __riscv_vmerge_vvm_f32m4(result, sign_x, mask_large, vl);

    vbool8_t mask_small = __riscv_vmfle_vf_f32m4_b8(abs_x, THRESHOLD, vl);
    if (__riscv_vfirst_m_b8(mask_small, vl) >= 0) {
        vfloat32m4_t x_small = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(0.0f, vl), x, mask_small, vl);
        vfloat32m4_t two_x = __riscv_vfmul_vf_f32m4(x_small, 2.0f, vl);
        vfloat32m4_t exp_2x = __gloden_vec_exp(two_x, vl);
        vfloat32m4_t numerator = __riscv_vfsub_vf_f32m4(exp_2x, 1.0f, vl);
        vfloat32m4_t denominator = __riscv_vfadd_vf_f32m4(exp_2x, 1.0f, vl);
        vfloat32m4_t tanh_x = __riscv_vfdiv_vv_f32m4(numerator, denominator, vl);
        result = __riscv_vmerge_vvm_f32m4(result, tanh_x, mask_small, vl);
    }
    return result;
}

/* ------------------------------------------------------------------ */
/*  vec_sin / vec_cos                                                 */
/* ------------------------------------------------------------------ */
static inline vfloat32m4_t __gloden_vec_sin_small(vfloat32m4_t x, size_t vl) {
    vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl);
    vfloat32m4_t c1  = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c3  = __riscv_vfmv_v_f_f32m4(-0.166666666667f, vl);
    vfloat32m4_t c5  = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c7  = __riscv_vfmv_v_f_f32m4(-0.0001984126984f, vl);
    vfloat32m4_t c9  = __riscv_vfmv_v_f_f32m4(0.000002755731922f, vl);
    vfloat32m4_t c11 = __riscv_vfmv_v_f_f32m4(-0.000000025052108f, vl);

    vfloat32m4_t result;
    result = __riscv_vfmacc_vv_f32m4(c9,     c11, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c7,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c5,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c3,  result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c1,  result, x2, vl);
    result = __riscv_vfmul_vv_f32m4(result, x, vl);
    return result;
}

static inline vfloat32m4_t __gloden_vec_sin(vfloat32m4_t x, size_t vl) {
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    vfloat32m4_t pi_vec = __riscv_vfmv_v_f_f32m4(PI, vl);
    vfloat32m4_t round = __riscv_vfdiv_vv_f32m4(new_rad, pi_vec, vl);
    vfloat32m4_t magic = __riscv_vfmv_v_f_f32m4(0x1.8p23f, vl);
    round = __riscv_vfadd_vv_f32m4(round, magic, vl);
    round = __riscv_vfsub_vv_f32m4(round, magic, vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmul_vv_f32m4(round, pi_vec, vl), vl);
    new_rad = __riscv_vfsub_vv_f32m4(new_rad, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);

    vfloat32m4_t sin_result = __gloden_vec_sin_small(new_rad, vl);

    vuint32m4_t round_int = __riscv_vfcvt_xu_f_v_u32m4(round, vl);
    vuint32m4_t round_odd_int = __riscv_vand_vx_u32m4(round_int, 1, vl);
    round_odd_int = __riscv_vsll_vx_u32m4(round_odd_int, 31, vl);
    vfloat32m4_t sign = __riscv_vreinterpret_v_u32m4_f32m4(round_odd_int);
    sin_result = __riscv_vfsgnjn_vv_f32m4(sin_result, sign, vl);
    return sin_result;
}

static inline vfloat32m4_t __gloden_vec_cos(vfloat32m4_t x, size_t vl) {
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    return __gloden_vec_sin(new_rad, vl);
}

/* ------------------------------------------------------------------ */
/*  RMSnorm                                                           */
/* ------------------------------------------------------------------ */
static inline void __gloden_RMSnorm(float* input, float* output, float* per_channel_scale,
                                     float rms_epsilon, int batch, int seq_len, int hidden_dim)
{
    assert(batch > 0 && seq_len > 0 && hidden_dim > 0);
    assert(hidden_dim % (16*4) == 0);

    for (int b = 0; b < batch; b++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0;
            size_t avl, vl;
            size_t vl_0 = __riscv_vsetvl_e32m4(hidden_dim);
            vfloat32m4_t sum_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t vec_2 = __riscv_vfmul_vv_f32m4(vec, vec, vl);
                sum_vec = __riscv_vfadd_vv_f32m4(sum_vec, vec_2, vl);
            }
            sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sum_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));
            float rms = 1.0f / fast_sqrt(sum / hidden_dim + rms_epsilon);
            vfloat32m4_t rms_vec = __riscv_vfmv_v_f_f32m4(rms, vl_0);
            for (int h = 0, avl = hidden_dim; avl > 0; h += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input[b * seq_len * hidden_dim + j * hidden_dim + h], vl);
                vfloat32m4_t per_channel_scale_vec = __riscv_vle32_v_f32m4(&per_channel_scale[h], vl);
                vfloat32m4_t scaled_vec = __riscv_vfmul_vv_f32m4(vec, rms_vec, vl);
                scaled_vec = __riscv_vfmul_vv_f32m4(scaled_vec, per_channel_scale_vec, vl);
                __riscv_vse32_v_f32m4(&output[b * seq_len * hidden_dim + j * hidden_dim + h], scaled_vec, vl);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  RoPE                                                              */
/* ------------------------------------------------------------------ */
static inline void __gloden_rope(float* input, float* output, float* rope_theta,
                                  int pos, int batch, int n_head, int seq_len, int head_dim)
{
    const int half_dim = head_dim / 2;
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < n_head; i++) {
            for (int j = 0; j < seq_len; j++) {
                int pos_ = j + pos;
                int input_offset = b * n_head * seq_len * head_dim +
                                  i * seq_len * head_dim + j * head_dim;
                int output_offset = input_offset;
                size_t avl, vl;
                for (int k = 0, avl = half_dim; avl > 0; k += vl, avl -= vl) {
                    vl = __riscv_vsetvl_e32m4(avl);
                    vfloat32m4_t theta_vec = __riscv_vle32_v_f32m4(rope_theta + k, vl);
                    vfloat32m4_t angle_vec = __riscv_vfmul_vf_f32m4(theta_vec, pos_, vl);

                    vfloat32m4_t sin_vec = __gloden_vec_sin(angle_vec, vl);
                    vfloat32m4_t cos_vec = __gloden_vec_cos(angle_vec, vl);

                    vfloat32m4_t real_in = __riscv_vlse32_v_f32m4(input + input_offset + 2 * k, 2*sizeof(float), vl);
                    vfloat32m4_t imag_in = __riscv_vlse32_v_f32m4(input + input_offset + 2 * k + 1, 2*sizeof(float), vl);

                    vfloat32m4_t real_out = __riscv_vfmsub_vv_f32m4(real_in, cos_vec,
                                                                   __riscv_vfmul_vv_f32m4(imag_in, sin_vec, vl), vl);
                    vfloat32m4_t imag_out = __riscv_vfmacc_vv_f32m4(__riscv_vfmul_vv_f32m4(real_in, sin_vec, vl),
                                                                   imag_in, cos_vec, vl);

                    __riscv_vse32_v_f32m4(output + output_offset + k, real_out, vl);
                    __riscv_vse32_v_f32m4(output + output_offset + half_dim + k, imag_out, vl);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  SiLU                                                              */
/* ------------------------------------------------------------------ */
static inline void __gloden_silu(float* input, float* output, int batch, int exhidden_dim, int hidden_dim)
{
    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < exhidden_dim; s++) {
            float* row_input  = &input[b * exhidden_dim * hidden_dim + s * hidden_dim];
            float* row_output = &output[b * exhidden_dim * hidden_dim + s * hidden_dim];
            size_t avl, vl;
            for (int i = 0, avl = hidden_dim; avl > 0; i += vl, avl -= vl) {
                vl = __riscv_vsetvl_e32m4(avl);
                vfloat32m4_t x = __riscv_vle32_v_f32m4(&row_input[i], vl);
                vfloat32m4_t exp_neg_x = __gloden_vec_exp(__riscv_vfneg_v_f32m4(x, vl), vl);
                vfloat32m4_t silu = __riscv_vfdiv_vv_f32m4(x, __riscv_vfadd_vf_f32m4(exp_neg_x, 1.0f, vl), vl);
                __riscv_vse32_v_f32m4(&row_output[i], silu, vl);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  SmoothQuant                                                       */
/* ------------------------------------------------------------------ */
static inline void __gloden_smoothquantO1_stage1_getscale(float* A, float* scale, int M, int K)
{
    assert(K % (64*4) == 0);
    assert(M % 16 == 0);
    assert(M <= 1024);
    assert(K <= 32768);

    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(K);
        vl = vl_0;
        vfloat32m4_t tmp = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
            tmp = __riscv_vfmax_vv_f32m4(tmp, vfabs, vl);
        }
        vfloat32m1_t tmp_m1_max = __riscv_vfmv_v_f_f32m1(0.0f, vl_0);
        tmp_m1_max = __riscv_vfredmax_vs_f32m4_f32m1(tmp, tmp_m1_max, vl_0);
        float token_max = __riscv_vfmv_f_s_f32m1_f32(tmp_m1_max);
        scale[i] = token_max / 127.0f;
    }
}

static inline void __gloden_smoothquantO1_stage2_quant(float* A, int8_t* output, float* scale, int M, int K)
{
    for (int i = 0; i < M; i++) {
        float* row_A = &A[i * K];
        int8_t* output_row = &output[i * K];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(K);
        vl = vl_0;
        float id = 1.0f / scale[i];
        for (int j = 0, avl = K; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t v_x = __riscv_vle32_v_f32m4(&row_A[j], vl);
            vfloat32m4_t x0  = __riscv_vfmul_vf_f32m4(v_x, id, vl);
            vint16m2_t   vi  = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
            vint8m1_t    vs  = __riscv_vncvt_x_x_w_i8m1(vi, vl);
            __riscv_vse8_v_i8m1(&output_row[j], vs, vl);
        }
    }
}

static inline void __gloden_smoothquantO1(float* A, int8_t* output, float* scale, int M, int K)
{
    __gloden_smoothquantO1_stage1_getscale(A, scale, M, K);
    __gloden_smoothquantO1_stage2_quant(A, output, scale, M, K);
}

/* ------------------------------------------------------------------ */
/*  cvrtfp16 (F32 -> _Float16)                                        */
/* ------------------------------------------------------------------ */
static inline void __gloden_cvrtfp16(void* x, void* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float* input_row_f32 = &((float*)x)[i*N];
        _Float16* output_row_f16 = &((_Float16*)y)[i*N];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);
        vl = vl_0;
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row_f32[j], vl);
            vfloat16m2_t vec_fp16 = __riscv_vfncvt_f_f_w_f16m2(vec, vl);
            __riscv_vse16_v_f16m2((_Float16*)&output_row_f16[j], vec_fp16, vl);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  cvrtbf16 (F32 -> BF16 truncation)                                 */
/* ------------------------------------------------------------------ */
static inline vuint16m2_t __gloden_f32m4_to_bf16m2_trunc(vfloat32m4_t input, size_t vl)
{
    vuint32m4_t bits = __riscv_vreinterpret_v_f32m4_u32m4(input);
    bits = __riscv_vsrl_vx_u32m4(bits, 16, vl);
    return __riscv_vnsrl_wx_u16m2(bits, 0, vl);
}

static inline void __gloden_cvrtbf16(void* x, void* y, int M, int N) {
    for (int i = 0; i < M; i++) {
        float* input_row_f32 = &((float*)x)[i*N];
        uint16_t* output_row_bf16 = &((uint16_t*)y)[i*N];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);
        vl = vl_0;
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row_f32[j], vl);
            vuint16m2_t vec_bf16 = __gloden_f32m4_to_bf16m2_trunc(vec, vl);
            __riscv_vse16_v_u16m2(&output_row_bf16[j], vec_bf16, vl);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  softmax (with bitmask mask)                                       */
/* ------------------------------------------------------------------ */
static inline void __gloden_softmax(float* x, float* y, void* bitmask_ptr, int M, int N)
{
    for (int i = 0; i < M; i++) {
        float* row_x = &x[i * N];
        float* row_y = &y[i * N];
        size_t avl, vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(N);

        float max_val = row_x[0];
        vfloat32m4_t max_vec = __riscv_vfmv_v_f_f32m4(max_val, vl_0);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            vbool8_t mask = __riscv_vlm_v_b8((uint8_t*)(bitmask_ptr + (i * N + j)/8), vl);
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-INFINITY, vl), vec, mask, vl);
            max_vec = __riscv_vfmax_vv_f32m4(max_vec, vec, vl);
        }
        max_val = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(max_vec, __riscv_vfmv_v_f_f32m1(-INFINITY, vl_0), vl_0));

        float sum_exp = 0.0f;
        vfloat32m4_t sumexp_vec = __riscv_vfmv_v_f_f32m4(0.0f, vl_0);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_x[j], vl);
            vec = __riscv_vfsub_vf_f32m4(vec, max_val, vl);
            vbool8_t mask = __riscv_vlm_v_b8((uint8_t*)(bitmask_ptr + (i * N + j)/8), vl);
            vec = __riscv_vmerge_vvm_f32m4(__riscv_vfmv_v_f_f32m4(-90, vl), vec, mask, vl);
            vfloat32m4_t exp_vec = __gloden_vec_exp(vec, vl);
            __riscv_vse32_v_f32m4(&row_y[j], exp_vec, vl);
            sumexp_vec = __riscv_vfadd_vv_f32m4(sumexp_vec, exp_vec, vl);
        }
        sum_exp = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(sumexp_vec, __riscv_vfmv_v_f_f32m1(0.0f, vl_0), vl_0));

        vfloat32m4_t inv_sum_exp_vec = __riscv_vfmv_v_f_f32m4(1.0f / sum_exp, vl_0);
        for (int j = 0, avl = N; avl > 0; j += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&row_y[j], vl);
            vfloat32m4_t normalized = __riscv_vfmul_vv_f32m4(vec, inv_sum_exp_vec, vl);
            __riscv_vse32_v_f32m4(&row_y[j], normalized, vl);
        }
    }
}

#endif /* NVWA_GLODEN_OPT_H */
