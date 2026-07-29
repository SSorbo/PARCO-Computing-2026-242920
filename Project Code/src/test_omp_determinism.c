#include "math_utils.h"
#include "math_omp.h"
#include "test_common.h"
#include "test_suites.h"
#include <omp.h>

/*
 * OpenMP kernel determinism verification uses three strategies:
 *
 * 1. DETERMINISM — run the same computation 20 times; all outputs must be
 *    bitwise-identical. Variation would indicate a race or another
 *    unintended shared-state dependency.
 *
 * 2. THREAD-COUNT SWEEP — request prime and non-prime team sizes. The tiled
 *    partitioning must stay numerically equivalent as threads divide the grid.
 *
 * 3. STRESS — use tiny matrices with many threads (more threads than tiles).
 *    This exercises the case where many team members receive no output tile.
 */

#define DETERM_REPEATS 20
#define STRESS_REPEATS 100
#define STRESS_SIZE    8

/* ---- helpers ---- */

static void run_standard_gemm(void (*gemm)(float*,float*,float*,int,int,int),
                               float* A, float* B, float* C,
                               int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    gemm(A, B, C, M, K, N);
}

static void run_transposeB_gemm(void (*gemm)(float*,float*,float*,int,int,int),
                                 float* A, float* B, float* C,
                                 int M, int K, int N) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    gemm(A, B, C, M, K, N);
}

static void run_transposeA_gemm(void (*gemm)(float*,float*,float*,int,int,int),
                                 float* A, float* B, float* C,
                                 int M, int K, int N) {
    memset(C, 0, (size_t)K * N * sizeof(float));
    gemm(A, B, C, M, K, N);
}

/* ---- 1. Determinism test ---- */

static int test_determinism_standard(const char* name,
                                      void (*gemm)(float*,float*,float*,int,int,int),
                                      int M, int K, int N) {
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(K * N);
    float* C_prev = aligned_alloc_float(M * N);
    float* C_cur  = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, K * N);

    run_standard_gemm(gemm, A, B, C_prev, M, K, N);

    for (int rep = 1; rep < DETERM_REPEATS; rep++) {
        run_standard_gemm(gemm, A, B, C_cur, M, K, N);
        if (!arrays_bitwise_equal(C_prev, C_cur, M * N)) {
            printf("  [%s] DETERMINISM FAIL at repetition %d (M=%d K=%d N=%d)\n",
                   name, rep, M, K, N);
            compare_arrays(C_prev, C_cur, M * N, TEST_TOLERANCE);
            free(A); free(B); free(C_prev); free(C_cur);
            return 1;
        }
        /* swap buffers for next comparison without copying */
        float* tmp = C_prev; C_prev = C_cur; C_cur = tmp;
    }

    free(A); free(B); free(C_prev); free(C_cur);
    return 0;
}

static int test_determinism_transposeB(const char* name,
                                        void (*gemm)(float*,float*,float*,int,int,int),
                                        int M, int K, int N) {
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(N * K);
    float* C_prev = aligned_alloc_float(M * N);
    float* C_cur  = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, N * K);

    run_transposeB_gemm(gemm, A, B, C_prev, M, K, N);

    for (int rep = 1; rep < DETERM_REPEATS; rep++) {
        run_transposeB_gemm(gemm, A, B, C_cur, M, K, N);
        if (!arrays_bitwise_equal(C_prev, C_cur, M * N)) {
            printf("  [%s] DETERMINISM FAIL at repetition %d (M=%d K=%d N=%d)\n",
                   name, rep, M, K, N);
            free(A); free(B); free(C_prev); free(C_cur);
            return 1;
        }
        float* tmp = C_prev; C_prev = C_cur; C_cur = tmp;
    }

    free(A); free(B); free(C_prev); free(C_cur);
    return 0;
}

static int test_determinism_transposeA(const char* name,
                                        void (*gemm)(float*,float*,float*,int,int,int),
                                        int M, int K, int N) {
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(M * N);
    float* C_prev = aligned_alloc_float(K * N);
    float* C_cur  = aligned_alloc_float(K * N);

    fill_random(A, M * K);
    fill_random(B, M * N);

    run_transposeA_gemm(gemm, A, B, C_prev, M, K, N);

    for (int rep = 1; rep < DETERM_REPEATS; rep++) {
        run_transposeA_gemm(gemm, A, B, C_cur, M, K, N);
        if (!arrays_bitwise_equal(C_prev, C_cur, K * N)) {
            printf("  [%s] DETERMINISM FAIL at repetition %d (M=%d K=%d N=%d)\n",
                   name, rep, M, K, N);
            free(A); free(B); free(C_prev); free(C_cur);
            return 1;
        }
        float* tmp = C_prev; C_prev = C_cur; C_cur = tmp;
    }

    free(A); free(B); free(C_prev); free(C_cur);
    return 0;
}

/* ---- 2. Thread-count sweep ---- */

static int test_thread_sweep_standard(const char* name,
                                       void (*gemm)(float*,float*,float*,int,int,int),
                                       int M, int K, int N) {
    const int thread_counts[] = {1, 2, 3, 4, 5, 7, 8, 12, 16};
    const int n_counts = sizeof(thread_counts) / sizeof(int);

    float* A   = aligned_alloc_float(M * K);
    float* B   = aligned_alloc_float(K * N);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_test = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, K * N);

    /* Reference: single-threaded */
    omp_set_num_threads(1);
    run_standard_gemm(gemm, A, B, C_ref, M, K, N);

    for (int t = 1; t < n_counts; t++) {
        omp_set_num_threads(thread_counts[t]);
        run_standard_gemm(gemm, A, B, C_test, M, K, N);
        if (compare_arrays(C_test, C_ref, M * N, TEST_TOLERANCE) > 0) {
            printf("  [%s] THREAD-SWEEP FAIL at OMP_NUM_THREADS=%d (M=%d K=%d N=%d)\n",
                   name, thread_counts[t], M, K, N);
            free(A); free(B); free(C_ref); free(C_test);
            return 1;
        }
    }

    free(A); free(B); free(C_ref); free(C_test);
    return 0;
}

static int test_thread_sweep_transposeB(const char* name,
                                         void (*gemm)(float*,float*,float*,int,int,int),
                                         int M, int K, int N) {
    const int thread_counts[] = {1, 2, 3, 4, 5, 7, 8, 12, 16};
    const int n_counts = sizeof(thread_counts) / sizeof(int);

    float* A   = aligned_alloc_float(M * K);
    float* B   = aligned_alloc_float(N * K);
    float* C_ref = aligned_alloc_float(M * N);
    float* C_test = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, N * K);

    omp_set_num_threads(1);
    run_transposeB_gemm(gemm, A, B, C_ref, M, K, N);

    for (int t = 1; t < n_counts; t++) {
        omp_set_num_threads(thread_counts[t]);
        run_transposeB_gemm(gemm, A, B, C_test, M, K, N);
        if (compare_arrays(C_test, C_ref, M * N, TEST_TOLERANCE) > 0) {
            printf("  [%s] THREAD-SWEEP FAIL at OMP_NUM_THREADS=%d (M=%d K=%d N=%d)\n",
                   name, thread_counts[t], M, K, N);
            free(A); free(B); free(C_ref); free(C_test);
            return 1;
        }
    }

    free(A); free(B); free(C_ref); free(C_test);
    return 0;
}

static int test_thread_sweep_transposeA(const char* name,
                                         void (*gemm)(float*,float*,float*,int,int,int),
                                         int M, int K, int N) {
    const int thread_counts[] = {1, 2, 3, 4, 5, 7, 8, 12, 16};
    const int n_counts = sizeof(thread_counts) / sizeof(int);

    float* A   = aligned_alloc_float(M * K);
    float* B   = aligned_alloc_float(M * N);
    float* C_ref = aligned_alloc_float(K * N);
    float* C_test = aligned_alloc_float(K * N);

    fill_random(A, M * K);
    fill_random(B, M * N);

    omp_set_num_threads(1);
    run_transposeA_gemm(gemm, A, B, C_ref, M, K, N);

    for (int t = 1; t < n_counts; t++) {
        omp_set_num_threads(thread_counts[t]);
        run_transposeA_gemm(gemm, A, B, C_test, M, K, N);
        if (compare_arrays(C_test, C_ref, K * N, TEST_TOLERANCE) > 0) {
            printf("  [%s] THREAD-SWEEP FAIL at OMP_NUM_THREADS=%d (M=%d K=%d N=%d)\n",
                   name, thread_counts[t], M, K, N);
            free(A); free(B); free(C_ref); free(C_test);
            return 1;
        }
    }

    free(A); free(B); free(C_ref); free(C_test);
    return 0;
}

/* ---- 3. Stress test (tiny matrices, many threads) ---- */

static int test_stress_standard(const char* name,
                                 void (*gemm)(float*,float*,float*,int,int,int)) {
    const int M = STRESS_SIZE, K = STRESS_SIZE, N = STRESS_SIZE;
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(K * N);
    float* C_prev = aligned_alloc_float(M * N);
    float* C_cur  = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, K * N);

    omp_set_num_threads(16);  /* more threads than tiles */
    run_standard_gemm(gemm, A, B, C_prev, M, K, N);

    for (int rep = 1; rep < STRESS_REPEATS; rep++) {
        run_standard_gemm(gemm, A, B, C_cur, M, K, N);
        if (!arrays_bitwise_equal(C_prev, C_cur, M * N)) {
            printf("  [%s] STRESS FAIL at repetition %d\n", name, rep);
            free(A); free(B); free(C_prev); free(C_cur);
            return 1;
        }
        float* tmp = C_prev; C_prev = C_cur; C_cur = tmp;
    }

    free(A); free(B); free(C_prev); free(C_cur);
    return 0;
}

static int test_stress_transposeB(const char* name,
                                   void (*gemm)(float*,float*,float*,int,int,int)) {
    const int M = STRESS_SIZE, K = STRESS_SIZE, N = STRESS_SIZE;
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(N * K);
    float* C_prev = aligned_alloc_float(M * N);
    float* C_cur  = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, N * K);

    omp_set_num_threads(16);
    run_transposeB_gemm(gemm, A, B, C_prev, M, K, N);

    for (int rep = 1; rep < STRESS_REPEATS; rep++) {
        run_transposeB_gemm(gemm, A, B, C_cur, M, K, N);
        if (!arrays_bitwise_equal(C_prev, C_cur, M * N)) {
            printf("  [%s] STRESS FAIL at repetition %d\n", name, rep);
            free(A); free(B); free(C_prev); free(C_cur);
            return 1;
        }
        float* tmp = C_prev; C_prev = C_cur; C_cur = tmp;
    }

    free(A); free(B); free(C_prev); free(C_cur);
    return 0;
}

static int test_stress_transposeA(const char* name,
                                   void (*gemm)(float*,float*,float*,int,int,int)) {
    const int M = STRESS_SIZE, K = STRESS_SIZE, N = STRESS_SIZE;
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(M * N);
    float* C_prev = aligned_alloc_float(K * N);
    float* C_cur  = aligned_alloc_float(K * N);

    fill_random(A, M * K);
    fill_random(B, M * N);

    omp_set_num_threads(16);
    run_transposeA_gemm(gemm, A, B, C_prev, M, K, N);

    for (int rep = 1; rep < STRESS_REPEATS; rep++) {
        run_transposeA_gemm(gemm, A, B, C_cur, M, K, N);
        if (!arrays_bitwise_equal(C_prev, C_cur, K * N)) {
            printf("  [%s] STRESS FAIL at repetition %d\n", name, rep);
            free(A); free(B); free(C_prev); free(C_cur);
            return 1;
        }
        float* tmp = C_prev; C_prev = C_cur; C_cur = tmp;
    }

    free(A); free(B); free(C_prev); free(C_cur);
    return 0;
}

/* ---- public entry point ---- */

int run_omp_kernel_tests(void) {
    int failed = 0;

    printf("\n=== OpenMP Kernel Determinism Tests ===\n");
    printf("Using up to %d OpenMP threads\n", omp_get_max_threads());

    const int M = 128, K = 128, N = 128;

    /* --- Determinism --- */
    printf("\n--- Determinism (%d repetitions, must be bitwise-identical) ---\n", DETERM_REPEATS);

    int before = failed;
    failed += test_determinism_standard("mat_mat_mult_tiled",
                                        mat_mat_mult_tiled, M, K, N);
    failed += test_determinism_transposeB("mat_mat_mult_tiled_transposeB",
                                           mat_mat_mult_tiled_transposeB, M, K, N);
    failed += test_determinism_transposeA("mat_mat_mult_tiled_transposeA",
                                           mat_mat_mult_tiled_transposeA, M, K, N);

    printf("  determinism: %s\n",
           failed == before ? "PASS" : "FAIL (see above)");

    /* --- Thread-count sweep --- */
    printf("\n--- Thread-count sweep (1,2,3,4,5,7,8,12,16 threads) ---\n");

    before = failed;
    failed += test_thread_sweep_standard("mat_mat_mult_tiled",
                                         mat_mat_mult_tiled, M, K, N);
    failed += test_thread_sweep_transposeB("mat_mat_mult_tiled_transposeB",
                                            mat_mat_mult_tiled_transposeB, M, K, N);
    failed += test_thread_sweep_transposeA("mat_mat_mult_tiled_transposeA",
                                            mat_mat_mult_tiled_transposeA, M, K, N);
    printf("  thread-count sweep: %s\n",
           failed == before ? "PASS" : "FAIL (see above)");

    /* --- Stress --- */
    printf("\n--- Stress (%d iterations, %dx%d matrices, 16 threads) ---\n",
           STRESS_REPEATS, STRESS_SIZE, STRESS_SIZE);

    before = failed;
    failed += test_stress_standard("mat_mat_mult_tiled", mat_mat_mult_tiled);
    failed += test_stress_transposeB("mat_mat_mult_tiled_transposeB",
                                      mat_mat_mult_tiled_transposeB);
    failed += test_stress_transposeA("mat_mat_mult_tiled_transposeA",
                                      mat_mat_mult_tiled_transposeA);

    printf("  stress: %s\n",
           failed == before ? "PASS" : "FAIL (see above)");

    int tests_run = 9;  /* three strategies applied to three kernels */
    printf("\nOpenMP kernel determinism: %d tests, %d failed\n",
           tests_run, failed);
    return failed;
}
