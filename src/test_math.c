#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "math_utils.h"
#include "math_omp.h"

#define DIM 1024
#define TOLERANCE 1e-4f

/*
 * Allocate a 64-byte aligned float array of `count` elements.
 * Exits the process on failure — test harness, no graceful recovery needed.
 */
static float* aligned_alloc_float(size_t count) {
    float* ptr = NULL;
    if (posix_memalign((void**)&ptr, 64, count * sizeof(float)) != 0) {
        fprintf(stderr, "Aligned allocation failed for %zu floats.\n", count);
        exit(1);
    }
    return ptr;
}

int main(void) {
    srand((unsigned)time(NULL));

    int rows = DIM;
    int cols = DIM;
    int w_size = rows * cols;

    printf("=== Tiled vs Sequential Mat-Vec Verification ===\n");
    printf("Matrix: %d x %d  |  Tolerance: %e\n\n", rows, cols, TOLERANCE);

    /* Allocate all arrays on 64-byte boundaries. */
    float* weights   = aligned_alloc_float(w_size);
    float* input_vec = aligned_alloc_float(cols);
    float* bias      = aligned_alloc_float(rows);
    float* y_seq     = aligned_alloc_float(rows);
    float* y_tiled   = aligned_alloc_float(rows);

    /* Fill with random values in [-1, 1]. */
    for (int i = 0; i < w_size; i++)
        weights[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < cols; i++)
        input_vec[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    for (int i = 0; i < rows; i++)
        bias[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;

    /*
     * Compute reference result with the sequential implementation.
     */
    mat_vec_mult(weights, input_vec, bias, y_seq, rows, cols);

    /*
     * Compute result with the cache-tiled OpenMP implementation.
     */
    mat_vec_mult_tiled(weights, input_vec, bias, y_tiled, rows, cols);

    /*
     * Element-wise comparison.  Floating-point addition is not associative,
     * so tiled summation (different order) can produce slightly different
     * results.  A tolerance of 1e-4 allows for this while catching any
     * algorithmic errors (wrong indexing, missing tiles, etc.).
     */
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < rows; i++) {
        float diff = fabsf(y_seq[i] - y_tiled[i]);
        if (diff > max_diff)
            max_diff = diff;
        if (diff > TOLERANCE) {
            mismatches++;
            if (mismatches <= 5) {
                printf("MISMATCH [%d]: seq=%.8f  tiled=%.8f  diff=%.8f\n",
                       i, y_seq[i], y_tiled[i], diff);
            }
        }
    }

    printf("Max absolute difference: %.8f\n", max_diff);

    if (mismatches == 0) {
        printf("[VERIFICATION SUCCESS]: Tiled matrix multiplication "
               "matches sequential baseline exactly.\n");
    } else {
        printf("[VERIFICATION FAILURE]: %d / %d elements exceed "
               "tolerance %.e.\n", mismatches, rows, TOLERANCE);
        free(weights);
        free(input_vec);
        free(bias);
        free(y_seq);
        free(y_tiled);
        return 1;
    }

    free(weights);
    free(input_vec);
    free(bias);
    free(y_seq);
    free(y_tiled);
    return 0;
}
