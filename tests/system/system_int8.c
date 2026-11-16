#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
#include "cute_checkpoint.h"
#include "system_inputs.h"

#ifndef CUTE_SYSTEM_FAULT
#define CUTE_SYSTEM_FAULT 0
#endif

/* Output symbols are the common task identities; the runner resolves each ELF. */
/* Loaded .data avoids spending simulated CPU cycles clearing large BSS arrays.
 * The verifier still requires a new successful store ack for every output byte. */
int8_t sys_a[128 * 128] __attribute__((aligned(64))) = {1};
int8_t sys_b[128 * 128] __attribute__((aligned(64))) = {-1};
int32_t sys_c[128 * 128] __attribute__((aligned(64))) = {0x13579bdf};
int32_t sys_v[128 * 128] __attribute__((aligned(64))) = {0x2468ace0};

static void copy_i8(int8_t *dst, const int8_t *src, size_t count)
{
    while (count) {
        size_t vl = __riscv_vsetvl_e8m4(count);
        vint8m4_t v = __riscv_vle8_v_i8m4(src, vl);
        __riscv_vse8_v_i8m4(dst, v, vl);
        src += vl;
        dst += vl;
        count -= vl;
    }
}

static void copy_i32(int32_t *dst, const int32_t *src, size_t count)
{
    while (count) {
        size_t vl = __riscv_vsetvl_e32m4(count);
        vint32m4_t v = __riscv_vle32_v_i32m4(src, vl);
        if (CUTE_SYSTEM_FAULT == 2) {
            v = __riscv_vadd_vx_i32m4(v, 1, vl);
        }
        __riscv_vse32_v_i32m4(dst, v, vl);
        src += vl;
        dst += vl;
        count -= vl;
    }
}

int main(void)
{
    cute_checkpoint_context context = {0};
    copy_i8(sys_a, sys_input_a, sizeof(sys_a));
    copy_i8(sys_b, sys_input_b, sizeof(sys_b));
    if (!cute_checkpoint(&context, 1)) {
        return 11;
    }

    uint64_t task = cute_matmul(sys_a, 128, sys_b, 128,
                                sys_c, 512, sys_c, 512,
                                128, 128, 128, 0, 1, 0, 0);
    if (CUTE_SYSTEM_FAULT != 1) {
        cute_wait_task(task);
    }
    if (!cute_checkpoint(&context, 2)) {
        return 12;
    }
    copy_i32(sys_v, sys_c, 128 * 128);
    if (!cute_checkpoint(&context, 3)) {
        return 13;
    }
    if (CUTE_SYSTEM_FAULT) {
        /* Keep the negative run alive so the online verifier must stop it. */
        for (;;) {
            (void)cute_checkpoint(&context, 3);
        }
    }
    cute_checkpoint_finish();
    return 0;
}
