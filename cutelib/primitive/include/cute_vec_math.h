#ifndef CUTE_VEC_MATH_H
#define CUTE_VEC_MATH_H

#include <math.h>
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

static inline float cute_fast_sqrt(float x)
{
    float result;
    __asm__ volatile ("fsqrt.s %0, %1" : "=f"(result) : "f"(x));
    return result;
}

static inline vfloat32m4_t cute_vec_exp(vfloat32m4_t x, size_t vl)
{
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
    p = __riscv_vfmacc_vv_f32m4(c4, p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c3, p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c2, p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c1, p, b, vl);
    p = __riscv_vfmacc_vv_f32m4(c0, p, b, vl);
    p = __riscv_vfmul_vv_f32m4(a2, p, vl);

    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(INFINITY, vl), mask_max, vl);
    p = __riscv_vmerge_vvm_f32m4(p, __riscv_vfmv_v_f_f32m4(0.0f, vl), mask_min, vl);
    return p;
}

static inline vfloat32m4_t cute_vec_sin_small(vfloat32m4_t x, size_t vl)
{
    vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl);
    vfloat32m4_t c1  = __riscv_vfmv_v_f_f32m4(1.0f, vl);
    vfloat32m4_t c3  = __riscv_vfmv_v_f_f32m4(-0.166666666667f, vl);
    vfloat32m4_t c5  = __riscv_vfmv_v_f_f32m4(0.008333333333f, vl);
    vfloat32m4_t c7  = __riscv_vfmv_v_f_f32m4(-0.0001984126984f, vl);
    vfloat32m4_t c9  = __riscv_vfmv_v_f_f32m4(0.000002755731922f, vl);
    vfloat32m4_t c11 = __riscv_vfmv_v_f_f32m4(-0.000000025052108f, vl);

    vfloat32m4_t result;
    result = __riscv_vfmacc_vv_f32m4(c9, c11, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c7, result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c5, result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c3, result, x2, vl);
    result = __riscv_vfmacc_vv_f32m4(c1, result, x2, vl);
    result = __riscv_vfmul_vv_f32m4(result, x, vl);
    return result;
}

static inline vfloat32m4_t cute_vec_sin(vfloat32m4_t x, size_t vl)
{
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

    vfloat32m4_t sin_result = cute_vec_sin_small(new_rad, vl);

    vuint32m4_t round_int = __riscv_vfcvt_xu_f_v_u32m4(round, vl);
    vuint32m4_t round_odd_int = __riscv_vand_vx_u32m4(round_int, 1, vl);
    round_odd_int = __riscv_vsll_vx_u32m4(round_odd_int, 31, vl);
    vfloat32m4_t sign = __riscv_vreinterpret_v_u32m4_f32m4(round_odd_int);
    sin_result = __riscv_vfsgnjn_vv_f32m4(sin_result, sign, vl);
    return sin_result;
}

static inline vfloat32m4_t cute_vec_cos(vfloat32m4_t x, size_t vl)
{
    const float PI = 3.14159265359f;
    const float PI_DIV_2 = PI / 2.0f;
    vfloat32m4_t new_rad = __riscv_vfadd_vv_f32m4(x, __riscv_vfmv_v_f_f32m4(PI_DIV_2, vl), vl);
    return cute_vec_sin(new_rad, vl);
}

#endif /* CUTE_VEC_MATH_H */
