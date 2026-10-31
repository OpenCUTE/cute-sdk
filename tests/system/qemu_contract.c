/* QEMU-only ISA/model checks. CPU reads here do not belong to the RTL workflow. */
#include <stdint.h>
#include "cute_runtime.h"

enum { M = 16, N = 32, K = 64, AS = 128, BS = 128, CS = 48, DS = 48 };
static int8_t a[M * AS] __attribute__((aligned(64)));
static int8_t b[N * BS] __attribute__((aligned(64)));
static uint32_t bias[M * CS] __attribute__((aligned(64)));
static uint32_t output[3][N * DS] __attribute__((aligned(64)));
static volatile unsigned traps, bad_trap;

__attribute__((interrupt("machine"), aligned(4))) static void illegal_handler(void)
{
    uintptr_t cause, epc;
    __asm__ volatile("csrr %0, mcause" : "=r"(cause));
    __asm__ volatile("csrr %0, mepc" : "=r"(epc));
    if (cause != 2) bad_trap = 1;
    ++traps;
    epc += 4; /* All tested custom-0 instructions are 32 bits. */
    __asm__ volatile("csrw mepc, %0" :: "r"(epc) : "memory");
}

int main(void)
{
    __asm__ volatile("csrw mtvec, %0" :: "r"(illegal_handler) : "memory");
    for (unsigned i = 0; i < M; ++i)
        for (unsigned k = 0; k < AS; ++k)
            a[i * AS + k] = k < K ? (i & 1 ? 127 : -128) : 63;
    for (unsigned j = 0; j < N; ++j)
        for (unsigned k = 0; k < BS; ++k)
            b[j * BS + k] = k < K ? (j & 1 ? -127 : 126) : -61;
    for (unsigned i = 0; i < M; ++i)
        for (unsigned j = 0; j < CS; ++j)
            bias[i * CS + j] = 0x7fffff00U + (i * N + j) * 17U;

    unsigned next_id = 0;
    for (unsigned round = 0; round < 8; ++round) {
        uint64_t ids[3], mask = 0;
        for (unsigned test = 0; test < 3; ++test) {
            for (unsigned i = 0; i < N * DS; ++i) output[test][i] = 0x12345678;
            ids[test] = cute_matmul(a, AS, b, BS, bias, test == 1 ? 0 : CS * 4,
                                    output[test], DS * 4, M, N, K,
                                    0, test + 1, test == 2, 0);
            if (ids[test] != next_id) return 30;
            mask |= 1ULL << next_id;
            next_id = (next_id + 1) & 3;
        }
        if (!cute_fifo_full() || cute_fifo_info() != mask || cute_query_finish() != mask) return 31;
        (void)CUTE_SEND_MACRO_INST(); /* Full queue must trap without changing it. */
        if (traps != round + 1 || bad_trap || cute_fifo_info() != mask) return 32;
        for (unsigned test = 0; test < 3; ++test) cute_wait_task(ids[test]);
        if (cute_fifo_full() || cute_fifo_info() || cute_query_finish()) return 33;
        for (unsigned test = 0; test < 3; ++test) {
            unsigned rows = test == 2 ? N : M;
            unsigned cols = test == 2 ? M : N;
            for (unsigned r = 0; r < N; ++r) {
                for (unsigned c = 0; c < DS; ++c) {
                    uint32_t expected = 0x12345678;
                    if (r < rows && c < cols) {
                        unsigned i = test == 2 ? c : r, j = test == 2 ? r : c;
                        int av = i & 1 ? 127 : -128, bv = j & 1 ? -127 : 126;
                        expected = (uint32_t)(K * av * bv); /* Closed-form dot product. */
                        if (test) expected += bias[(test == 1 ? 0 : i) * CS + j];
                    }
                    if (output[test][r * DS + c] != expected) return 34;
                }
            }
        }
    }
    uint64_t unused;
    YGJK_INS_RRR(unused, 0, 0, 127);
    (void)unused;
    CUTE_CONFIG_CONV_PARAMS(1, 1, 0, 1, 0, 0, 1, 0, 64, 0, 0);
    (void)CUTE_SEND_MACRO_INST(); /* CUTE FP16 is explicitly not implemented yet. */
    return traps == 10 && !bad_trap && cute_fifo_info() == 0 ? 0 : 35;
}
