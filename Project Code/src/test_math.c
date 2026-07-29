#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "math_utils.h"
#include "math_omp.h"

#define TOLERANCE 1e-4f

/* Allocate a 64-byte aligned float array of `count` elements. */
static float* aligned_alloc_float(size_t count) {
    float* ptr = NULL;
    if (posix_memalign((void**)&ptr, 64, count * sizeof(float)) != 0) {
        fprintf(stderr, "Aligned allocation failed for %zu floats.\n", count);
        exit(1);
    }
    return ptr;
}

static void fill_random(float* data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
}

int test_mat_mat_mult(int M, int K, int N) {
    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(K * N);
    float* C_seq = aligned_alloc_float(M * N);
    float* C_tiled = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, K * N);
    memset(C_seq, 0, M * N * sizeof(float));
    memset(C_tiled, 0, M * N * sizeof(float));

    mat_mat_mult(A, B, C_seq, M, K, N);

    mat_mat_mult_tiled(A, B, C_tiled, M, K, N);
    for (int i = 0; i < M * N; i++) {
        if (fabsf(C_seq[i] - C_tiled[i]) > TOLERANCE) {
            printf("[GEMM FAILURE] M=%d K=%d N=%d index=%d seq=%.6f tiled=%.6f\n", 
                   M, K, N, i, C_seq[i], C_tiled[i]);
            free(A); free(B); free(C_seq); free(C_tiled);
            return 1;
        }
    }
    free(A); free(B); free(C_seq); free(C_tiled);
    return 0;
}

int test_mat_mat_mult_transposeB(int M, int K, int N) {
    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(N * K);
    float* C_seq = aligned_alloc_float(M * N);
    float* C_tiled = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, N * K);
    memset(C_seq, 0, M * N * sizeof(float));
    memset(C_tiled, 0, M * N * sizeof(float));

    mat_mat_mult_transposeB(A, B, C_seq, M, K, N);

    mat_mat_mult_tiled_transposeB(A, B, C_tiled, M, K, N);

    for (int i = 0; i < M * N; i++) {
        if (fabsf(C_seq[i] - C_tiled[i]) > TOLERANCE) {
            printf("[GEMM_TB FAILURE] M=%d K=%d N=%d index=%d seq=%.6f tiled=%.6f\n",
                   M, K, N, i, C_seq[i], C_tiled[i]);
            free(A); free(B); free(C_seq); free(C_tiled);
            return 1;
        }
    }
    free(A); free(B); free(C_seq); free(C_tiled);
    return 0;
}

int test_mat_mat_mult_transposeA(int M, int K, int N) {
    float* A = aligned_alloc_float(M * K);
    float* B = aligned_alloc_float(M * N);
    float* C_seq = aligned_alloc_float(K * N);
    float* C_tiled = aligned_alloc_float(K * N);

    fill_random(A, M * K);
    fill_random(B, M * N);
    memset(C_seq, 0, K * N * sizeof(float));
    memset(C_tiled, 0, K * N * sizeof(float));

    mat_mat_mult_transposeA(A, B, C_seq, M, K, N);

    mat_mat_mult_tiled_transposeA(A, B, C_tiled, M, K, N);

    for (int i = 0; i < K * N; i++) {
        if (fabsf(C_seq[i] - C_tiled[i]) > TOLERANCE) {
            printf("[GEMM_TA FAILURE] M=%d K=%d N=%d index=%d seq=%.6f tiled=%.6f\n",
                   M, K, N, i, C_seq[i], C_tiled[i]);
            free(A); free(B); free(C_seq); free(C_tiled);
            return 1;
        }
    }
    free(A); free(B); free(C_seq); free(C_tiled);
    return 0;
}

int main(void) {
    srand((unsigned)time(NULL));

    int passed = 0;
    int failed = 0;

    /* Large square matrix */
    if (test_mat_mat_mult(1024, 1024, 1024) == 0) { passed++; } else { failed++; }

    if (test_mat_mat_mult(64, 64, 64) == 0) passed++; else failed++;   // Perfect block
    if (test_mat_mat_mult(70, 70, 70) == 0) passed++; else failed++;   // Partial block

    if (test_mat_mat_mult_transposeB(64, 64, 64) == 0) passed++; else failed++;   // Perfect block
    if (test_mat_mat_mult_transposeB(70, 70, 70) == 0) passed++; else failed++;   // Partial block

    if (test_mat_mat_mult_transposeA(64, 64, 64) == 0) passed++; else failed++;   // Perfect block
    if (test_mat_mat_mult_transposeA(70, 70, 70) == 0) passed++; else failed++;   // Partial block
    
    if (failed > 0) {
        printf("[TEST SUITE FAILED] %d failed, %d passed.\n", failed, passed);
        return 1;
    }

    printf("[TEST SUITE PASSED ALL CHECKS] %d passed.\n", passed);
    return 0;
}
