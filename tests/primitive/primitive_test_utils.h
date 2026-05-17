#ifndef CUTE_PRIMITIVE_TEST_UTILS_H
#define CUTE_PRIMITIVE_TEST_UTILS_H

#include <stddef.h>
#include <string.h>

#define CUTE_TEST_ALIGN __attribute__((aligned(64)))
#define CUTE_TEST_BLOCK 64

static inline void cute_test_host_verify_barrier(void)
{
    __asm__ volatile ("fence rw, rw" ::: "memory");
}

static inline int cute_check_bytes(const void *actual, const void *expected, size_t bytes)
{
    return memcmp(actual, expected, bytes) == 0 ? 0 : 1;
}

static inline size_t cute_test_min_size(size_t lhs, size_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

static inline int cute_check_matrix_element(const unsigned char *actual,
                                            const unsigned char *expected,
                                            size_t cols,
                                            size_t row,
                                            size_t col,
                                            size_t elem_bytes)
{
    const size_t offset = (row * cols + col) * elem_bytes;
    for (size_t i = 0; i < elem_bytes; ++i) {
        if (actual[offset + i] != expected[offset + i]) {
            return 1;
        }
    }
    return 0;
}

static inline int cute_check_star64_blocks(const void *actual,
                                           const void *expected,
                                           size_t rows,
                                           size_t cols,
                                           size_t elem_bytes)
{
    const unsigned char *actual_bytes = (const unsigned char *)actual;
    const unsigned char *expected_bytes = (const unsigned char *)expected;

    if (rows == 0 || cols == 0 || elem_bytes == 0) {
        return 0;
    }

    for (size_t block_row = 0; block_row < rows; block_row += CUTE_TEST_BLOCK) {
        const size_t block_rows = cute_test_min_size(CUTE_TEST_BLOCK, rows - block_row);
        for (size_t block_col = 0; block_col < cols; block_col += CUTE_TEST_BLOCK) {
            const size_t block_cols = cute_test_min_size(CUTE_TEST_BLOCK, cols - block_col);
            const size_t diag_len = cute_test_min_size(block_rows, block_cols);
            const size_t mid_row = block_row + block_rows / 2;
            const size_t mid_col = block_col + block_cols / 2;

            for (size_t i = 0; i < diag_len; ++i) {
                if (cute_check_matrix_element(actual_bytes,
                                              expected_bytes,
                                              cols,
                                              block_row + i,
                                              block_col + i,
                                              elem_bytes)) {
                    return 1;
                }
                if (cute_check_matrix_element(actual_bytes,
                                              expected_bytes,
                                              cols,
                                              block_row + i,
                                              block_col + block_cols - 1 - i,
                                              elem_bytes)) {
                    return 1;
                }
            }

            for (size_t col = 0; col < block_cols; ++col) {
                if (cute_check_matrix_element(actual_bytes,
                                              expected_bytes,
                                              cols,
                                              mid_row,
                                              block_col + col,
                                              elem_bytes)) {
                    return 1;
                }
            }

            for (size_t row = 0; row < block_rows; ++row) {
                if (cute_check_matrix_element(actual_bytes,
                                              expected_bytes,
                                              cols,
                                              block_row + row,
                                              mid_col,
                                              elem_bytes)) {
                    return 1;
                }
            }
        }
    }

    return 0;
}

#endif /* CUTE_PRIMITIVE_TEST_UTILS_H */
