#include <stdint.h>
#include <riscv_vector.h>
#include "cute_checkpoint.h"
#include "system_inputs.h"
#include "system_platform.h"

#define CUTE_STRING_1(value) #value
#define CUTE_STRING(value) CUTE_STRING_1(value)
/* Absolute output symbol: no ELF load/BSS initialization is assumed for TCM. */
__asm__(".globl sys_tcm_output\n"
        ".type sys_tcm_output, @object\n"
        ".set sys_tcm_output, " CUTE_STRING(CUTE_SYSTEM_TCM_BASE) "\n"
        ".size sys_tcm_output, 64\n");
extern int8_t sys_tcm_output[64];
int8_t sys_tcm_readback[64] __attribute__((aligned(64)));

int main(void)
{
    cute_checkpoint_context context = {0};
    size_t vl = __riscv_vsetvl_e8m1(64);
    if (vl != 64) return 20;
    vint8m1_t input = __riscv_vle8_v_i8m1(sys_input_a, vl);
    __riscv_vse8_v_i8m1(sys_tcm_output, input, vl);
    if (!cute_checkpoint(&context, 1)) return 21;
    vint8m1_t result = __riscv_vle8_v_i8m1(sys_tcm_output, vl);
    __riscv_vse8_v_i8m1(sys_tcm_readback, result, vl);
    if (!cute_checkpoint(&context, 2)) return 22;
    cute_checkpoint_finish();
    return 0;
}
