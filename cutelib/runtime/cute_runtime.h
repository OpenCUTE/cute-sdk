#ifndef CUTE_RUNTIME_H
#define CUTE_RUNTIME_H

#include "instruction.h"

// SCP tile dimensions (hardware-specific)
#define CUTE_SCP_M 64
#define CUTE_SCP_N 64

static inline uint64_t cute_matmul(
    const void *a, uint64_t a_stride,
    const void *b, uint64_t b_stride,
    void *bias, uint64_t bias_stride,
    void *d, uint64_t d_stride,
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t element_type,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t m_index)
{
    CUTE_CONFIG_TENSOR_A((uint64_t)a, a_stride);
    CUTE_CONFIG_TENSOR_B((uint64_t)b, b_stride);
    CUTE_CONFIG_TENSOR_C((uint64_t)bias, bias_stride);
    CUTE_CONFIG_TENSOR_D((uint64_t)d, d_stride);
    CUTE_CONFIG_TENSOR_DIM(m, n, k, 0);
    CUTE_CONFIG_CONV_PARAMS(
        element_type, bias_mode, transpose,
        1, 0, 0,           /* conv_stride=1, oh_max=0, ow_max=0 */
        1,                  /* kernel_size=1 */
        0, CUTE_SCP_M,     /* oh_per_add=0, ow_per_add=SCP_M */
        0, m_index          /* oh_index=0, ow_index=m_index */
    );
    return CUTE_SEND_MACRO_INST();
}

static inline uint64_t cute_blockscale_matmul(
    const void *a, uint64_t a_stride,
    const void *b, uint64_t b_stride,
    const void *scale_a, const void *scale_b,
    void *bias, uint64_t bias_stride,
    void *d, uint64_t d_stride,
    uint64_t m, uint64_t n, uint64_t k,
    uint64_t element_type,
    uint64_t bias_mode,
    uint64_t transpose,
    uint64_t m_index)
{
    CUTE_CONFIG_TENSOR_A((uint64_t)a, a_stride);
    CUTE_CONFIG_TENSOR_B((uint64_t)b, b_stride);
    CUTE_CONFIG_SCALE_A((uint64_t)scale_a);
    CUTE_CONFIG_SCALE_B((uint64_t)scale_b);
    CUTE_CONFIG_TENSOR_C((uint64_t)bias, bias_stride);
    CUTE_CONFIG_TENSOR_D((uint64_t)d, d_stride);
    CUTE_CONFIG_TENSOR_DIM(m, n, k, 0);
    CUTE_CONFIG_CONV_PARAMS(
        element_type, bias_mode, transpose,
        1, 0, 0,
        1,
        0, CUTE_SCP_M,
        0, m_index
    );
    return CUTE_SEND_MACRO_INST();
}

#endif /* CUTE_RUNTIME_H */
