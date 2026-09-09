// 连续 GEMM 宏任务回归。
//
// 单任务程序每次提交一条 RoCC 宏指令后立即等待完成，无法覆盖 FIFO 占用、
// opcode/scale/bias 切换及多个 D 输出区域同时存活的情况。本程序把八份
// generated header 链接进同一个 RISC-V ELF，并按 task_order 连续提交。
// 每份 header 的符号均重命名，确保 A/B/C/D/scale 与 golden 完全隔离。
#include <stdint.h>
#include <stdio.h>
#include "cute_runtime.h"

// 非 scale dtype 不生成 a_scale/b_scale。对应 CaseDesc 的 scale 地址置零，
// 此占位数组仅保留为调试时可读的安全地址。
static int8_t unused_scale[1] __attribute__((aligned(256)));

#define a a0
#define b b0
#define c c0
#define d d0
#define gloden_c gloden_c0
#define a_scale a_scale0
#define b_scale b_scale0
#include "generated/batch_case_0.h"
enum { M0 = APPLICATION_M, N0 = APPLICATION_N, K0 = APPLICATION_K, T0 = ELEMENT_TYPE, B0 = BIAS_TYPE, X0 = TRANSPOSE_RESULT, S0 = HAS_SCALE, SA0 = STRIDE_A, SB0 = STRIDE_B, SC0 = STRIDE_C, SD0 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a1
#define b b1
#define c c1
#define d d1
#define gloden_c gloden_c1
#define a_scale a_scale1
#define b_scale b_scale1
#include "generated/batch_case_1.h"
enum { M1 = APPLICATION_M, N1 = APPLICATION_N, K1 = APPLICATION_K, T1 = ELEMENT_TYPE, B1 = BIAS_TYPE, X1 = TRANSPOSE_RESULT, S1 = HAS_SCALE, SA1 = STRIDE_A, SB1 = STRIDE_B, SC1 = STRIDE_C, SD1 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a2
#define b b2
#define c c2
#define d d2
#define gloden_c gloden_c2
#define a_scale a_scale2
#define b_scale b_scale2
#include "generated/batch_case_2.h"
enum { M2 = APPLICATION_M, N2 = APPLICATION_N, K2 = APPLICATION_K, T2 = ELEMENT_TYPE, B2 = BIAS_TYPE, X2 = TRANSPOSE_RESULT, S2 = HAS_SCALE, SA2 = STRIDE_A, SB2 = STRIDE_B, SC2 = STRIDE_C, SD2 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a3
#define b b3
#define c c3
#define d d3
#define gloden_c gloden_c3
#define a_scale a_scale3
#define b_scale b_scale3
#include "generated/batch_case_3.h"
enum { M3 = APPLICATION_M, N3 = APPLICATION_N, K3 = APPLICATION_K, T3 = ELEMENT_TYPE, B3 = BIAS_TYPE, X3 = TRANSPOSE_RESULT, S3 = HAS_SCALE, SA3 = STRIDE_A, SB3 = STRIDE_B, SC3 = STRIDE_C, SD3 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a4
#define b b4
#define c c4
#define d d4
#define gloden_c gloden_c4
#define a_scale a_scale4
#define b_scale b_scale4
#include "generated/batch_case_4.h"
enum { M4 = APPLICATION_M, N4 = APPLICATION_N, K4 = APPLICATION_K, T4 = ELEMENT_TYPE, B4 = BIAS_TYPE, X4 = TRANSPOSE_RESULT, S4 = HAS_SCALE, SA4 = STRIDE_A, SB4 = STRIDE_B, SC4 = STRIDE_C, SD4 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a5
#define b b5
#define c c5
#define d d5
#define gloden_c gloden_c5
#define a_scale a_scale5
#define b_scale b_scale5
#include "generated/batch_case_5.h"
enum { M5 = APPLICATION_M, N5 = APPLICATION_N, K5 = APPLICATION_K, T5 = ELEMENT_TYPE, B5 = BIAS_TYPE, X5 = TRANSPOSE_RESULT, S5 = HAS_SCALE, SA5 = STRIDE_A, SB5 = STRIDE_B, SC5 = STRIDE_C, SD5 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a6
#define b b6
#define c c6
#define d d6
#define gloden_c gloden_c6
#define a_scale a_scale6
#define b_scale b_scale6
#include "generated/batch_case_6.h"
enum { M6 = APPLICATION_M, N6 = APPLICATION_N, K6 = APPLICATION_K, T6 = ELEMENT_TYPE, B6 = BIAS_TYPE, X6 = TRANSPOSE_RESULT, S6 = HAS_SCALE, SA6 = STRIDE_A, SB6 = STRIDE_B, SC6 = STRIDE_C, SD6 = STRIDE_D };
#undef a
#undef b
#undef c
#undef d
#undef gloden_c
#undef a_scale
#undef b_scale
#undef APPLICATION_M
#undef APPLICATION_N
#undef APPLICATION_K
#undef ELEMENT_TYPE
#undef CMODEL_DTYPE
#undef GOLDEN_SEED
#undef FPE_BYTES
#undef HAS_SCALE
#undef SCALE_COUNT
#undef BIAS_TYPE
#undef STRIDE_A
#undef STRIDE_B
#undef STRIDE_C
#undef STRIDE_D
#undef TRANSPOSE_RESULT

#define a a7
#define b b7
#define c c7
#define d d7
#define gloden_c gloden_c7
#define a_scale a_scale7
#define b_scale b_scale7
#include "generated/batch_case_7.h"
enum { M7 = APPLICATION_M, N7 = APPLICATION_N, K7 = APPLICATION_K, T7 = ELEMENT_TYPE, B7 = BIAS_TYPE, X7 = TRANSPOSE_RESULT, S7 = HAS_SCALE, SA7 = STRIDE_A, SB7 = STRIDE_B, SC7 = STRIDE_C, SD7 = STRIDE_D };

typedef struct {
    // 前六项是 RISC-V 虚拟地址；后续字段直接映射到 CUTE 宏指令的 stride、
    // M/N/K、opcode、bias_type、transpose_result 和 scale 使能位。
    uint64_t a, b, a_scale, b_scale, c, d;
    uint64_t sa, sb, sc, sd, m, n, k, dtype, bias, transpose, has_scale;
} BatchCase;

#ifndef BATCH_MODE
#define BATCH_MODE 0
#endif

#define DESC(i) { (uint64_t)a##i, (uint64_t)b##i, (uint64_t)a_scale##i, (uint64_t)b_scale##i, (uint64_t)c##i, (uint64_t)d##i, SA##i, SB##i, SC##i, SD##i, M##i, N##i, K##i, T##i, B##i, X##i, S##i }
// 对无 scale 的 case，a_scale/b_scale 名称未生成，单独使用零地址。
static BatchCase cases[8] = {
    { (uint64_t)a0, (uint64_t)b0, 0, 0, (uint64_t)c0, (uint64_t)d0, SA0, SB0, SC0, SD0, M0, N0, K0, T0, B0, X0, S0 },
    DESC(1), DESC(2),
    { (uint64_t)a3, (uint64_t)b3, 0, 0, (uint64_t)c3, (uint64_t)d3, SA3, SB3, SC3, SD3, M3, N3, K3, T3, B3, X3, S3 },
    DESC(4),
    { (uint64_t)a5, (uint64_t)b5, 0, 0, (uint64_t)c5, (uint64_t)d5, SA5, SB5, SC5, SD5, M5, N5, K5, T5, B5, X5, S5 },
    DESC(6),
    { (uint64_t)a7, (uint64_t)b7, 0, 0, (uint64_t)c7, (uint64_t)d7, SA7, SB7, SC7, SD7, M7, N7, K7, T7, B7, X7, S7 },
};

// 顺序实验只改变 task_order，不改变任何 case 的输入或 golden；因此 case 1
// NVFP4 的数值变化只能归因于前序任务，而不是随机数序列。
#if BATCH_MODE == 0
static const int task_order[] = {0, 1, 2, 3, 4, 5, 6, 7};
#elif BATCH_MODE == 1
static const int task_order[] = {1};
#elif BATCH_MODE == 2
static const int task_order[] = {0, 1};
#elif BATCH_MODE == 3
static const int task_order[] = {5, 1};
#elif BATCH_MODE == 4
static const int task_order[] = {3, 1};
#elif BATCH_MODE == 5
static const int task_order[] = {2, 1};
#else
#error "Unsupported BATCH_MODE"
#endif

static void issue_case(const BatchCase* item) {
    // scale dtype 必须走 blockscale 接口；普通 dtype 走普通接口。两种接口的
    // C/D 参数顺序均为“C/bias 输入在前，D 输出在后”。
    if (item->has_scale) {
        cute_blockscale_matmul(
            (const void *)(uintptr_t)item->a, item->sa,
            (const void *)(uintptr_t)item->b, item->sb,
            (const void *)(uintptr_t)item->a_scale,
            (const void *)(uintptr_t)item->b_scale,
            (void *)(uintptr_t)item->c, item->sc,
            (void *)(uintptr_t)item->d, item->sd, item->m, item->n, item->k,
            item->dtype, item->bias, item->transpose, 0);
    } else {
        cute_matmul(
            (const void *)(uintptr_t)item->a, item->sa,
            (const void *)(uintptr_t)item->b, item->sb,
            (void *)(uintptr_t)item->c, item->sc,
            (void *)(uintptr_t)item->d, item->sd, item->m, item->n, item->k,
            item->dtype, item->bias, item->transpose, 0);
    }
}

int main(void) {
    const int total = sizeof(task_order) / sizeof(task_order[0]);
    for (int slot = 0; slot < total; ++slot) {
        // 本配置的 TaskCtrl_AutoClear 默认关闭。FIFO 满时，必须等待队尾任务
        // 完成并显式 dequeue；否则完成任务仍占据 FIFO 槽位，后续指令会被丢弃。
        while (cute_fifo_full()) {
            while (!cute_query_finish()) {
            }
            cute_fifo_dequeue();
        }
        const BatchCase* item = &cases[task_order[slot]];
        printf("BATCH_D_RANGE case=%d base=%lx bytes=%lu\n", slot, item->d,
               item->m * item->n * 4UL);
        issue_case(item);
    }
    // drain 阶段逐条等待完成并清除队尾；valid=0 才表示全部八条任务结束。
    while (cute_fifo_info()) {
        while (!cute_query_finish()) {
        }
        cute_fifo_dequeue();
    }
    printf("SV FPE C-model GEMM batch: %d macro tasks completed\n", total);
    return 0;
}
