#ifndef CUTE_CHECKPOINT_H
#define CUTE_CHECKPOINT_H

#include "cute_runtime.h"

/* Each hart owns its context. Output descriptions belong to the task manifest. */
typedef struct {
    uint64_t sequence;
} cute_checkpoint_context;

/* Leave two heartbeat periods after the final checkpoint. The host requires
 * every port's complete-cycle counter after that checkpoint before accepting EOF.
 * This is a short CSR wait, not an output readback or a host polling delay. */
#ifndef CUTE_CHECKPOINT_DRAIN_CYCLES
#define CUTE_CHECKPOINT_DRAIN_CYCLES 512
#endif

static inline void cute_checkpoint_finish(void)
{
    uint64_t start, now;
    __asm__ volatile("fence rw,rw\n\trdcycle %0" : "=r"(start) :: "memory");
    do {
        __asm__ volatile("rdcycle %0" : "=r"(now) :: "memory");
    } while ((uint64_t)(now - start) < CUTE_CHECKPOINT_DRAIN_CYCLES);
}

static inline uint64_t cute_checkpoint(cute_checkpoint_context *context,
                                       uint64_t task_key)
{
    __asm__ volatile("fence rw,rw" ::: "memory");
    uint64_t sequence = ++context->sequence;
    /* The generated primitive consumes the RoCC result with a dependent add. */
    uint64_t accepted = CUTE_TRACE_MARK(sequence, task_key);
    __asm__ volatile("fence rw,rw" ::: "memory");
    return accepted == sequence;
}

#endif
