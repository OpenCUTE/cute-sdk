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

/* ---- FIFO 状态查询（非阻塞） ---- */

// 返回已完成任务的 bitmask，bit N = 1 表示 task_id=N 的指令已完成
static inline uint64_t cute_query_finish(void) {
    return CUTE_QUERY_MACRO_INST_FINISH();
}

// FIFO 是否已满（1=满，0=未满）
static inline int cute_fifo_full(void) {
    return CUTE_QUERY_MACRO_INST_FIFO_FULL() != 0;
}

// FIFO 当前占用状态（bitmask，bit N = 1 表示位置 N 有指令）
static inline uint64_t cute_fifo_info(void) {
    return CUTE_QUERY_MACRO_INST_FIFO_INFO();
}

/* ---- 等待与出队 ---- */

// 阻塞等待特定 task_id 完成，然后出队队首
//
// FIFO 顺序约束：
//   1. wait+dequeue 严格按 issue 顺序调用（不能跳过中间 task）
//   2. task_id 必须是当前 FIFO 中最早进入的那条未完成指令
//   3. 先 issue 的必定先完成，不存在乱序完成
static inline void cute_wait_task(uint64_t task_id) {
    uint64_t mask = 1UL << task_id;
    while (!(CUTE_QUERY_MACRO_INST_FINISH() & mask))
        ;
    CUTE_CLEAR_INST();
}

/* ---- 位置查询 ---- */

// 返回已完成宏指令的尾编号位置
static inline uint64_t cute_query_inst_tail(void) {
    return CUTE_QUERY_INST();
}

/* ---- 加速器状态 ---- */

// 对外访存读次数
static inline uint64_t cute_query_mem_read_count(void) {
    return CUTE_QUERY_MEM_READ_COUNT();
}

// 对外访存写次数
static inline uint64_t cute_query_mem_write_count(void) {
    return CUTE_QUERY_MEM_WRITE_COUNT();
}


#endif /* CUTE_RUNTIME_H */
