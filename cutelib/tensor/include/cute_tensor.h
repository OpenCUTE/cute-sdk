#ifndef CUTE_TENSOR_H
#define CUTE_TENSOR_H

#include <stdint.h>
#include "cute_fpe.h"

/* ---- Bias 加载模式 ---- */
#define CUTE_BIAS_ZERO        1
#define CUTE_BIAS_ROW_REPEAT  2
#define CUTE_BIAS_FULL        3

/* ---- Scale 类型 ---- */
#define CUTE_SCALE_NONE                   0
#define CUTE_SCALE_PERTOKEN_A_PERTENSOR_B 1

/* ---- 硬件 tile 尺寸 ---- */
#define CUTE_TILE_M 64
#define CUTE_TILE_N 64

/* ---- Tensor 描述符 ---- */
typedef struct {
    void    *data;
    uint64_t stride;    /* 行步长 (byte) */
    uint64_t rows;      /* M */
    uint64_t cols;      /* N 或 K */
    uint64_t dtype;     /* ElementDataType */
} cute_tensor_t;

/* ---- 辅助：计算输入 tensor 行步长 ---- */
static inline uint64_t cute_stride(uint64_t cols, uint64_t dtype) {
    uint64_t elem_bytes;
    switch (dtype) {
        case CUTEDataTypeI8I8I32:
        case CUTEDataTypeI8U8I32:
        case CUTEDataTypeU8I8I32:
        case CUTEDataTypeU8U8I32:
            elem_bytes = 1; break;
        case CUTEDataTypeF16F16F32:
        case CUTEDataTypeBF16BF16F32:
            elem_bytes = 2; break;
        case CUTEDataTypeTF32TF32F32:
            elem_bytes = 4; break;
        default:
            elem_bytes = 1; break;
    }
    return cols * elem_bytes;
}

/* 输出 tensor stride（始终为 I32 或 F32，4 byte） */
static inline uint64_t cute_output_stride(uint64_t cols) {
    return cols * 4;
}

#endif /* CUTE_TENSOR_H */
