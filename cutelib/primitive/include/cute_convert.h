#ifndef CUTE_CONVERT_H
#define CUTE_CONVERT_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

static inline float cute_dequant_i32(int32_t acc, float input_scale, float weight_scale)
{
    return (float)acc * input_scale * weight_scale;
}

static inline void cute_f32_to_f16_tile(const float *input, uint64_t input_stride,
                                        void *output, uint64_t output_stride,
                                        int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const float *input_row = (const float *)((const char *)input + r * input_stride);
        _Float16 *output_row = (_Float16 *)((char *)output + r * output_stride);
        size_t vl;
        size_t vl_0 = __riscv_vsetvl_e32m4(cols);
        vl = vl_0;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vfloat32m4_t vec = __riscv_vle32_v_f32m4(&input_row[c], vl);
            vfloat16m2_t vec_f16 = __riscv_vfncvt_f_f_w_f16m2(vec, vl);
            __riscv_vse16_v_f16m2(&output_row[c], vec_f16, vl);
        }
    }
}

static inline void cute_f32_to_bf16_tile(const float *input, uint64_t input_stride,
                                         void *output, uint64_t output_stride,
                                         int rows, int cols)
{
    cute_f32_to_f16_tile(input, input_stride, output, output_stride, rows, cols);
}

static inline void cute_dequant_i32_to_f32_tile(const int32_t *input, uint64_t input_stride,
                                                float *output, uint64_t output_stride,
                                                const float *input_scale,
                                                const float *weight_scale,
                                                int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float scale = input_scale[r] * weight_scale[0];
        const int32_t *input_row = (const int32_t *)((const char *)input + r * input_stride);
        float *output_row = (float *)((char *)output + r * output_stride);
        size_t vl;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(&input_row[c], vl);
            vfloat32m4_t deq = __riscv_vfmul_vf_f32m4(
                __riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);
            __riscv_vse32_v_f32m4(&output_row[c], deq, vl);
        }
    }
}

static inline void cute_dequant_i32_to_f16_tile(const int32_t *input, uint64_t input_stride,
                                                void *output, uint64_t output_stride,
                                                const float *input_scale,
                                                const float *weight_scale,
                                                int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float scale = input_scale[r] * weight_scale[0];
        const int32_t *input_row = (const int32_t *)((const char *)input + r * input_stride);
        _Float16 *output_row = (_Float16 *)((char *)output + r * output_stride);
        size_t vl;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(&input_row[c], vl);
            vfloat32m4_t deq = __riscv_vfmul_vf_f32m4(
                __riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale, vl);
            vfloat16m2_t deq_f16 = __riscv_vfncvt_f_f_w_f16m2(deq, vl);
            __riscv_vse16_v_f16m2(&output_row[c], deq_f16, vl);
        }
    }
}

static inline void cute_dequant_i32_to_f16_transpose_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        const int32_t *input_row = (const int32_t *)((const char *)input + r * input_stride);
        _Float16 *output_row = (_Float16 *)((char *)output + r * output_stride);
        size_t vl;
        for (int c = 0, avl = cols; avl > 0; c += vl, avl -= vl) {
            vl = __riscv_vsetvl_e32m4(avl);
            vint32m4_t input_vec = __riscv_vle32_v_i32m4(&input_row[c], vl);
            vfloat32m4_t scale_vec = __riscv_vle32_v_f32m4(&input_scale[c], vl);
            scale_vec = __riscv_vfmul_vf_f32m4(scale_vec, weight_scale[0], vl);
            vfloat32m4_t deq = __riscv_vfmul_vv_f32m4(
                __riscv_vfcvt_f_x_v_f32m4(input_vec, vl), scale_vec, vl);
            vfloat16m2_t deq_f16 = __riscv_vfncvt_f_f_w_f16m2(deq, vl);
            __riscv_vse16_v_f16m2(&output_row[c], deq_f16, vl);
        }
    }
}

static inline void cute_dequant_i32_to_bf16_tile(const int32_t *input, uint64_t input_stride,
                                                 void *output, uint64_t output_stride,
                                                 const float *input_scale,
                                                 const float *weight_scale,
                                                 int rows, int cols)
{
    cute_dequant_i32_to_f16_tile(input, input_stride, output, output_stride,
                                 input_scale, weight_scale, rows, cols);
}

static inline void cute_dequant_i32_to_bf16_transpose_tile(
    const int32_t *input, uint64_t input_stride,
    void *output, uint64_t output_stride,
    const float *input_scale,
    const float *weight_scale,
    int rows, int cols)
{
    cute_dequant_i32_to_f16_transpose_tile(input, input_stride,
                                           output, output_stride,
                                           input_scale, weight_scale,
                                           rows, cols);
}

#endif /* CUTE_CONVERT_H */
