#include "math_utils.h"
#include "math_omp.h"
#include "test_common.h"
#include "test_suites.h"

/* ---- naive reference implementations for standard and transposeA GEMM ---- */

static void naive_gemm_standard(const float* A, const float* B, float* C,
                                 int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        int a_row = i * K;
        int c_row = i * N;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[a_row + k] * B[k * N + j];
            }
            C[c_row + j] += sum;
        }
    }
}

static void naive_gemm_transposeA(const float* A, const float* B, float* C,
                                   int M, int K, int N) {
    for (int i = 0; i < K; i++) {
        int c_row = i * N;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += A[k * K + i] * B[k * N + j];
            }
            C[c_row + j] += sum;
        }
    }
}

/* ---- GEMM test: standard C += A * B ---- */

static int test_gemm_standard_one(int M, int K, int N) {
    float* A   = aligned_alloc_float(M * K);
    float* B   = aligned_alloc_float(K * N);
    float* C_ref   = aligned_alloc_float(M * N);
    float* C_seq   = aligned_alloc_float(M * N);
    float* C_tiled = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, K * N);
    memset(C_ref,   0, M * N * sizeof(float));
    memset(C_seq,   0, M * N * sizeof(float));
    memset(C_tiled, 0, M * N * sizeof(float));

    naive_gemm_standard(A, B, C_ref,   M, K, N);
    mat_mat_mult(A, B, C_seq,           M, K, N);
    mat_mat_mult_tiled(A, B, C_tiled,   M, K, N);

    int fail = 0;
    if (compare_arrays(C_seq, C_ref, M * N, TEST_TOLERANCE) > 0) {
        printf("  [STANDARD] seq_tiled vs naive FAIL (M=%d K=%d N=%d)\n", M, K, N);
        fail = 1;
    }
    if (compare_arrays(C_tiled, C_ref, M * N, TEST_TOLERANCE) > 0) {
        printf("  [STANDARD] omp_tiled vs naive FAIL (M=%d K=%d N=%d)\n", M, K, N);
        fail = 1;
    }

    free(A); free(B); free(C_ref); free(C_seq); free(C_tiled);
    return fail;
}

/* ---- GEMM test: C += A * B^T ---- */

static int test_gemm_transposeB_one(int M, int K, int N) {
    float* A      = aligned_alloc_float(M * K);
    float* B      = aligned_alloc_float(N * K);  /* B is N×K, we compute A × B^T */
    float* C_ref  = aligned_alloc_float(M * N);
    float* C_seq  = aligned_alloc_float(M * N);
    float* C_tiled = aligned_alloc_float(M * N);

    fill_random(A, M * K);
    fill_random(B, N * K);
    memset(C_ref,  0, M * N * sizeof(float));
    memset(C_seq,  0, M * N * sizeof(float));
    memset(C_tiled,0, M * N * sizeof(float));

    mat_mat_mult_naive(A, B, C_ref,               M, K, N);
    mat_mat_mult_transposeB(A, B, C_seq,           M, K, N);
    mat_mat_mult_tiled_transposeB(A, B, C_tiled,   M, K, N);

    int fail = 0;
    if (compare_arrays(C_seq, C_ref, M * N, TEST_TOLERANCE) > 0) {
        printf("  [TRANSPOSE_B] seq_tiled vs naive FAIL (M=%d K=%d N=%d)\n", M, K, N);
        fail = 1;
    }
    if (compare_arrays(C_tiled, C_ref, M * N, TEST_TOLERANCE) > 0) {
        printf("  [TRANSPOSE_B] omp_tiled vs naive FAIL (M=%d K=%d N=%d)\n", M, K, N);
        fail = 1;
    }

    free(A); free(B); free(C_ref); free(C_seq); free(C_tiled);
    return fail;
}

/* ---- GEMM test: C += A^T * B ---- */

static int test_gemm_transposeA_one(int M, int K, int N) {
    float* A      = aligned_alloc_float(M * K);   /* A is M×K, we compute A^T × B */
    float* B      = aligned_alloc_float(M * N);
    float* C_ref  = aligned_alloc_float(K * N);
    float* C_seq  = aligned_alloc_float(K * N);
    float* C_tiled = aligned_alloc_float(K * N);

    fill_random(A, M * K);
    fill_random(B, M * N);
    memset(C_ref,  0, K * N * sizeof(float));
    memset(C_seq,  0, K * N * sizeof(float));
    memset(C_tiled,0, K * N * sizeof(float));

    naive_gemm_transposeA(A, B, C_ref,            M, K, N);
    mat_mat_mult_transposeA(A, B, C_seq,           M, K, N);
    mat_mat_mult_tiled_transposeA(A, B, C_tiled,   M, K, N);

    int fail = 0;
    if (compare_arrays(C_seq, C_ref, K * N, TEST_TOLERANCE) > 0) {
        printf("  [TRANSPOSE_A] seq_tiled vs naive FAIL (M=%d K=%d N=%d)\n", M, K, N);
        fail = 1;
    }
    if (compare_arrays(C_tiled, C_ref, K * N, TEST_TOLERANCE) > 0) {
        printf("  [TRANSPOSE_A] omp_tiled vs naive FAIL (M=%d K=%d N=%d)\n", M, K, N);
        fail = 1;
    }

    free(A); free(B); free(C_ref); free(C_seq); free(C_tiled);
    return fail;
}

/* ---- test case tables ---- */

typedef struct { int M, K, N; } Dims;

static const Dims gemm_sizes[] = {
    /* Perfect block multiples */
    {64, 64, 64}, {128, 64, 128}, {64, 128, 64},
    /* Block-edge: one dimension just over a block boundary */
    {65, 64, 64}, {64, 65, 64}, {64, 64, 65},
    {127, 128, 128}, {129, 128, 128},
    /* Prime dimensions — no clean block alignment */
    {61, 67, 59}, {97, 89, 101}, {37, 53, 71},
    /* Odd dimensions */
    {63, 65, 63}, {127, 129, 131},
    /* Tiny — stress small-dimension paths */
    {1, 1, 1}, {1, 64, 1}, {1, 1, 64}, {3, 5, 7},
    /* Large inner dimension — stressful for cache */
    {64, 1024, 64}, {32, 2048, 32},
    /* Small inner dimension — tiling inner loop edge */
    {64, 1, 64}, {64, 2, 64}, {64, 3, 64},
    /* Large M × N */
    {256, 64, 256}, {512, 64, 512},
};

static const int gemm_count = sizeof(gemm_sizes) / sizeof(Dims);

/* ---- activation function tests ---- */

static int test_sigmoid(void) {
    int fail = 0;

    /* Known values */
    float s0 = sigmoid(0.0f);
    if (fabsf(s0 - 0.5f) > 1e-6f) {
        printf("  sigmoid(0) = %.8f, expected 0.5\n", s0);
        fail++;
    }

    /* Monotonicity */
    float prev = sigmoid(-10.0f);
    for (float x = -9.5f; x <= 10.0f; x += 0.5f) {
        float cur = sigmoid(x);
        if (cur < prev) {
            printf("  sigmoid not monotonic at x=%.1f\n", x);
            fail++;
            break;
        }
        prev = cur;
    }

    /* Symmetry: sigmoid(-x) = 1 - sigmoid(x) */
    for (float x = 0.0f; x <= 5.0f; x += 0.25f) {
        float diff = fabsf(sigmoid(-x) - (1.0f - sigmoid(x)));
        if (diff > 1e-6f) {
            printf("  sigmoid symmetry fail at x=%.2f: %.8f vs %.8f\n",
                   x, sigmoid(-x), 1.0f - sigmoid(x));
            fail++;
            break;
        }
    }

    /* Bounds: sigmoid(x) ∈ (0, 1) for values where float32 can represent it.
     * At |x| > ~17, expf(|x|) exceeds 2^23, so 1/(1+expf(-x)) rounds to 0 or 1. */
    for (float x = -10.0f; x <= 10.0f; x += 1.0f) {
        float s = sigmoid(x);
        if (s <= 0.0f || s >= 1.0f) {
            printf("  sigmoid(%.1f) = %.8f out of bounds\n", x, s);
            fail++;
        }
    }
    /* Edge: at |x|=16, float32 still just distinguishes sigmoid from 1.0.
     * sigmoid(16) ≈ 0.99999989 → rounds to 0.99999994 (nextafter(1.0, 0)).
     * At |x|≥17, expf(|x|) > 2^23 and sigmoid rounds to exactly 1.0 or 0.0. */
    { float s16 = sigmoid(16.0f); if (s16 <= 0.0f || s16 >= 1.0f) {
        printf("  sigmoid(16) = %.8f out of bounds\n", s16); fail++; }}
    { float sn16 = sigmoid(-16.0f); if (sn16 <= 0.0f || sn16 >= 1.0f) {
        printf("  sigmoid(-16) = %.8f out of bounds\n", sn16); fail++; }}

    return fail;
}

static int test_sigmoid_deriv(void) {
    int fail = 0;

    /* Consistency: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) */
    for (float x = -5.0f; x <= 5.0f; x += 0.25f) {
        float analytical = sigmoid_deriv(x);
        float from_sigmoid = sigmoid(x) * (1.0f - sigmoid(x));
        if (fabsf(analytical - from_sigmoid) > 1e-6f) {
            printf("  sigmoid_deriv fail at x=%.2f: %.8f vs %.8f\n",
                   x, analytical, from_sigmoid);
            fail++;
            break;
        }
    }

    /* Derivative is symmetric: sigmoid'(x) = sigmoid'(-x) */
    for (float x = 0.0f; x <= 5.0f; x += 0.5f) {
        if (fabsf(sigmoid_deriv(x) - sigmoid_deriv(-x)) > 1e-6f) {
            printf("  sigmoid_deriv symmetry fail at x=%.2f\n", x);
            fail++;
            break;
        }
    }

    /* Derivative at 0 should be 0.25 (max of sigmoid') */
    float d0 = sigmoid_deriv(0.0f);
    if (fabsf(d0 - 0.25f) > 1e-6f) {
        printf("  sigmoid_deriv(0) = %.8f, expected 0.25\n", d0);
        fail++;
    }

    return fail;
}

/* ---- public entry point ---- */

int run_correctness_tests(void) {
    int passed = 0, failed = 0;
    int total_ge = 0;

    printf("\n=== Numerical Correctness Tests ===\n");

    /* GEMM standard: seq_tiled and omp_tiled vs naive reference */
    printf("\n--- GEMM standard: C += A * B ---\n");
    total_ge = gemm_count * 2;  /* 2 comparisons per size */
    for (int i = 0; i < gemm_count; i++) {
        int f = test_gemm_standard_one(gemm_sizes[i].M, gemm_sizes[i].K, gemm_sizes[i].N);
        failed += f;
        passed += (2 - f);
    }
    printf("  %d/%d passed\n", passed, total_ge);

    /* GEMM transposeB: naive, seq_tiled, omp_tiled all agree */
    printf("\n--- GEMM C += A * B^T ---\n");
    int prev_passed = passed;
    total_ge = gemm_count * 2;
    for (int i = 0; i < gemm_count; i++) {
        int f = test_gemm_transposeB_one(gemm_sizes[i].M, gemm_sizes[i].K, gemm_sizes[i].N);
        failed += f;
        passed += (2 - f);
    }
    printf("  %d/%d passed\n", passed - prev_passed, total_ge);

    /* GEMM transposeA: seq_tiled and omp_tiled vs naive reference */
    printf("\n--- GEMM C += A^T * B ---\n");
    prev_passed = passed;
    total_ge = gemm_count * 2;
    for (int i = 0; i < gemm_count; i++) {
        int f = test_gemm_transposeA_one(gemm_sizes[i].M, gemm_sizes[i].K, gemm_sizes[i].N);
        failed += f;
        passed += (2 - f);
    }
    printf("  %d/%d passed\n", passed - prev_passed, total_ge);

    /* Activation functions */
    printf("\n--- Activation functions ---\n");

    int sf = test_sigmoid();
    int sdf = test_sigmoid_deriv();
    failed += sf + sdf;
    passed += 2 - ((sf > 0) + (sdf > 0));
    printf("  sigmoid:        %s\n", sf   == 0 ? "PASS" : "FAIL");
    printf("  sigmoid_deriv:  %s\n", sdf  == 0 ? "PASS" : "FAIL");

    printf("\nNumerical correctness: %d passed, %d failed\n", passed, failed);
    return failed;
}
