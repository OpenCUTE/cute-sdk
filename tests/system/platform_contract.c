/* QEMU-only target configuration probe, separate from RTL trace verification. */
#include <stdint.h>
#include <stddef.h>
#include "system_platform.h"

extern void *_sbrk(ptrdiff_t increment);

int main(void)
{
    uintptr_t bytes;
    __asm__ volatile("csrr %0, vlenb" : "=r"(bytes));
    if (bytes * 8 != CUTE_SYSTEM_VLEN) return 40;
    volatile uint64_t *last = (volatile uint64_t *)(uintptr_t)
        (CUTE_SYSTEM_PROGRAM_RAM_END - 64);
    *last = UINT64_C(0x1234aabbccdd5678);
    if (*last != UINT64_C(0x1234aabbccdd5678)) return 41;
#if CUTE_SYSTEM_TCM_BYTES
    volatile uint64_t *tcm = (volatile uint64_t *)(uintptr_t)CUTE_SYSTEM_TCM_BASE;
    tcm[0] = UINT64_C(0x1020304050607080);
    tcm[CUTE_SYSTEM_TCM_BYTES / 8 - 1] = UINT64_C(0x8877665544332211);
    if (tcm[0] != UINT64_C(0x1020304050607080) ||
        tcm[CUTE_SYSTEM_TCM_BYTES / 8 - 1] != UINT64_C(0x8877665544332211)) return 42;
#endif
    if (_sbrk(CUTE_SYSTEM_HEAP_BYTES + 4096) != (void *)-1) return 43;
    return 0;
}
