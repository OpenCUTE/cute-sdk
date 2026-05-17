/*
 * Deterministic input generators for RVV golden test cases.
 * Uses fixed xorshift32 PRNG — no libc rand() dependency.
 */

#ifndef CUTE_GOLDEN_INPUTS_H
#define CUTE_GOLDEN_INPUTS_H

#include <stdint.h>
#include <math.h>
#include <string.h>

static uint32_t cute_rng_state = 0x43555445u; /* "CUTE" */

static inline uint32_t cute_xorshift32(void) {
    uint32_t x = cute_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    cute_rng_state = x;
    return x;
}

static inline void cute_rng_seed(uint32_t seed) {
    cute_rng_state = seed;
}

/* uniform [lo, hi] F32 */
static inline float cute_rand_f32(float lo, float hi) {
    float t = (float)(cute_xorshift32() & 0xFFFFFF) / (float)0xFFFFFF;
    return lo + t * (hi - lo);
}

/* uniform [lo, hi] I32 */
static inline int32_t cute_rand_i32(int32_t lo, int32_t hi) {
    return lo + (int32_t)((uint64_t)cute_xorshift32() * (uint64_t)(hi - lo) >> 32);
}

static inline void cute_fill_random_f32(float *buf, int n, float lo, float hi) {
    for (int i = 0; i < n; i++)
        buf[i] = cute_rand_f32(lo, hi);
}

static inline void cute_fill_random_i32(int32_t *buf, int n, int32_t lo, int32_t hi) {
    for (int i = 0; i < n; i++)
        buf[i] = cute_rand_i32(lo, hi);
}

/* ---- per-op input generators ---- */

static inline void cute_fill_silu_input(float *buf, int n) {
    cute_rng_seed(0x43555445u);
    int kp = 0;
    float keypoints[] = {-30.0f, -10.0f, -5.0f, -1.0f, 0.0f, 1.0f, 5.0f, 10.0f, 30.0f};
    int nk = sizeof(keypoints) / sizeof(keypoints[0]);
    for (int i = 0; i < n; i++) {
        if (kp < nk && i % (n / nk) == 0)
            buf[i] = keypoints[kp++];
        else
            buf[i] = cute_rand_f32(-30.0f, 30.0f);
    }
}

static inline void cute_fill_vec_math_input(float *buf, int n) {
    cute_rng_seed(0x43555445u);
    /* exp keypoints: [-90, -20, -1, 0, 1, 20, 80] */
    /* sin/cos keypoints: [-8pi, -pi, -pi/2, 0, pi/2, pi, 8pi] */
    /* merge into one set of keypoints */
    float pi = 3.14159265359f;
    float keypoints[] = {
        -90.0f, -25.1327f, -20.0f, -3.14159f, -1.5708f,
        -1.0f, 0.0f, 1.0f, 1.5708f, 3.14159f,
        20.0f, 25.1327f, 80.0f
    };
    int nk = sizeof(keypoints) / sizeof(keypoints[0]);
    int kp = 0;
    for (int i = 0; i < n; i++) {
        if (kp < nk && i % (n / nk) == 0)
            buf[i] = keypoints[kp++];
        else
            buf[i] = cute_rand_f32(-90.0f, 80.0f);
    }
}

static inline void cute_fill_dequant_input_i32(int32_t *buf, int n) {
    cute_rng_seed(0x43555445u);
    for (int i = 0; i < n; i++)
        buf[i] = cute_rand_i32(-10000, 10000);
}

static inline void cute_fill_dequant_scale(float *input_scale, float *weight_scale, int M) {
    cute_rng_seed(0x5343414Cu); /* "SCAL" */
    for (int i = 0; i < M; i++)
        input_scale[i] = cute_rand_f32(0.001f, 0.1f);
    weight_scale[0] = cute_rand_f32(0.001f, 0.1f);
}

static inline void cute_fill_resadd_input(float *lhs, float *rhs, int n) {
    cute_rng_seed(0x43555445u);
    cute_fill_random_f32(lhs, n, -10.0f, 10.0f);
    cute_fill_random_f32(rhs, n, -10.0f, 10.0f);
}

static inline void cute_fill_hadamard_input(float *lhs, float *rhs, int n) {
    cute_rng_seed(0x43555445u);
    cute_fill_random_f32(lhs, n, -5.0f, 5.0f);
    cute_fill_random_f32(rhs, n, -5.0f, 5.0f);
}

static inline void cute_fill_rope_input(float *buf, int n) {
    cute_rng_seed(0x43555445u);
    cute_fill_random_f32(buf, n, -1.0f, 1.0f);
}

static inline void cute_fill_rope_theta(float *theta, int half_dim) {
    int head_dim = half_dim * 2;
    for (int k = 0; k < half_dim; k++)
        theta[k] = powf(10000.0f, -2.0f * (float)k / (float)head_dim);
}

static inline void cute_fill_causal_mask(uint8_t *mask, int M, int N) {
    /* causal: mask[i][j] = 1 if j <= i, else 0 */
    memset(mask, 0, (M * N + 7) / 8);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j <= i && j < N; j++) {
            int bit_idx = i * N + j;
            mask[bit_idx / 8] |= (1u << (bit_idx % 8));
        }
    }
}

static inline void cute_fill_softmax_input(float *buf, int n) {
    cute_rng_seed(0x43555445u);
    for (int i = 0; i < n; i++)
        buf[i] = cute_rand_f32(-5.0f, 5.0f);
}

static inline void cute_fill_smoothquant_input(float *buf, int n) {
    cute_rng_seed(0x43555445u);
    for (int i = 0; i < n; i++)
        buf[i] = cute_rand_f32(-50.0f, 50.0f);
}

static inline void cute_fill_rmsnorm_input(float *buf, int n) {
    cute_rng_seed(0x43555445u);
    cute_fill_random_f32(buf, n, -5.0f, 5.0f);
}

static inline void cute_fill_rmsnorm_weight(float *buf, int n) {
    cute_rng_seed(0x57454947u); /* "WEIG" */
    cute_fill_random_f32(buf, n, -1.0f, 1.0f);
}

#endif /* CUTE_GOLDEN_INPUTS_H */
