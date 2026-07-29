#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* 64-byte aligned allocation. Exits on failure (tests can't proceed without memory). */
static inline float* aligned_alloc_float(size_t count) {
    float* ptr = NULL;
    if (posix_memalign((void**)&ptr, 64, count * sizeof(float)) != 0) {
        fprintf(stderr, "FATAL: aligned allocation failed for %zu floats\n", count);
        exit(1);
    }
    return ptr;
}

/* Fill with uniform random in [-1, 1]. */
static inline void fill_random(float* data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

/* Fill with uniform random in [0, 1]. */
static inline void fill_random_positive(float* data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = (float)rand() / (float)RAND_MAX;
}

/* Compare two float arrays element-wise. Returns number of elements exceeding tol. */
static inline int compare_arrays(const float* a, const float* b, int n, float tol) {
    int failures = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            if (failures < 5)
                printf("  MISMATCH[%d]: %.8f vs %.8f (diff=%.2e)\n",
                       i, a[i], b[i], fabsf(a[i] - b[i]));
            failures++;
        }
    }
    if (failures > 5) printf("  ... and %d more mismatches\n", failures - 5);
    return failures;
}

/* Check if two float arrays are bitwise-identical (via memcmp). */
static inline int arrays_bitwise_equal(const float* a, const float* b, int n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

#define TEST_TOLERANCE       1e-4f
#define TEST_TOLERANCE_LOOSE 1e-3f   /* for non-associative parallel reductions */

#endif
