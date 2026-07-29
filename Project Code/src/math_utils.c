#include "math_utils.h"
#include <math.h>
#include <stdlib.h>

#define BLOCK_SIZE 64

static void mat_mat_mult_transposeA_naive(float *A, float *B, float *C,
                                          int M, int K, int N) {
    for (int i = 0; i < K; i++) {
        int c_row = i * N;
        for (int k = 0; k < M; k++) {
            float a_val = A[k * K + i];
            int b_row = k * N;
            for (int j = 0; j < N; j++)
                C[c_row + j] += a_val * B[b_row + j];
        }
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

void mat_mat_mult(float* A, float* B, float* C,
                  int M, int K, int N) {
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            int i_end = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
            int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                int k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;
                for (int i = ii; i < i_end; i++) {
                    int a_row = i * K;
                    int c_row = i * N;
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[a_row + k];
                        int b_row = k * N;
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

void mat_mat_mult_naive(float* A, float* B, float* C,
                        int M, int K, int N) {
    /*
     * C += A * B^T   where A is MxK, B is NxK, and C is MxN.
     * This direct O(M*K*N) triple loop uses no tiling or packed transpose.
     * Each individual dot product reads contiguous rows of A and B, but its
     * scalar k-inner loop does not expose the contiguous j-wise updates used
     * by the packed and tiled kernels.
     * This serves as the performance floor baseline.
     */
    for (int i = 0; i < M; i++) {
        int a_row = i * K;
        int c_row = i * N;
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[a_row + k] * B[j * K + k];
            }
            C[c_row + j] += sum;
        }
    }
}

void mat_mat_mult_transposeB(float* A, float* B, float* C,
                             int M, int K, int N) {
    /* Reused across calls; this routine is not safe for concurrent callers. */
    static float *B_T = NULL;
    static size_t B_T_cap = 0;
    size_t need = (size_t)K * N * sizeof(float);

    if (need > B_T_cap) {
        free(B_T);
        B_T = NULL;
        B_T_cap = 0;
        if (posix_memalign((void**)&B_T, 64, need) != 0) {
            mat_mat_mult_naive(A, B, C, M, K, N);
            return;
        }
        B_T_cap = need;
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B_T[k * N + j] = B[j * K + k];
        }
    }

    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            int i_end = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
            int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                int k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;
                for (int i = ii; i < i_end; i++) {
                    int a_row = i * K;
                    int c_row = i * N;
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[a_row + k];
                        int bt_row = k * N;
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B_T[bt_row + j];
                        }
                    }
                }
            }
        }
    }
}

void mat_mat_mult_transposeA(float* A, float* B, float* C,
                             int M, int K, int N) {
    /* Reused across calls; this routine is not safe for concurrent callers. */
    static float *A_T = NULL;
    static size_t A_T_cap = 0;
    size_t need = (size_t)K * M * sizeof(float);

    if (need > A_T_cap) {
        free(A_T);
        A_T = NULL;
        A_T_cap = 0;
        if (posix_memalign((void**)&A_T, 64, need) != 0) {
            mat_mat_mult_transposeA_naive(A, B, C, M, K, N);
            return;
        }
        A_T_cap = need;
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A_T[k * M + i] = A[i * K + k];
        }
    }

    for (int ii = 0; ii < K; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            int i_end = (ii + BLOCK_SIZE < K) ? ii + BLOCK_SIZE : K;
            int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
            for (int kk = 0; kk < M; kk += BLOCK_SIZE) {
                int k_end = (kk + BLOCK_SIZE < M) ? kk + BLOCK_SIZE : M;
                for (int i = ii; i < i_end; i++) {
                    int at_row = i * M;
                    int c_row = i * N;
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A_T[at_row + k];
                        int b_row = k * N;
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

/* ---- Runtime-tile-size variants (for tile-size sweep experiments) ---- */

void mat_mat_mult_rtile(float* A, float* B, float* C,
                        int M, int K, int N, int tile_size) {
    if (tile_size <= 0)
        tile_size = BLOCK_SIZE;
    for (int ii = 0; ii < M; ii += tile_size) {
        for (int jj = 0; jj < N; jj += tile_size) {
            int i_end = (ii + tile_size < M) ? ii + tile_size : M;
            int j_end = (jj + tile_size < N) ? jj + tile_size : N;
            for (int kk = 0; kk < K; kk += tile_size) {
                int k_end = (kk + tile_size < K) ? kk + tile_size : K;
                for (int i = ii; i < i_end; i++) {
                    int a_row = i * K;
                    int c_row = i * N;
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[a_row + k];
                        int b_row = k * N;
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

void mat_mat_mult_transposeB_rtile(float* A, float* B, float* C,
                                   int M, int K, int N, int tile_size) {
    if (tile_size <= 0)
        tile_size = BLOCK_SIZE;
    /* Reused across calls; this routine is not safe for concurrent callers. */
    static float *B_T = NULL;
    static size_t B_T_cap = 0;
    size_t need = (size_t)K * N * sizeof(float);

    if (need > B_T_cap) {
        free(B_T);
        B_T = NULL;
        B_T_cap = 0;
        if (posix_memalign((void**)&B_T, 64, need) != 0) {
            mat_mat_mult_naive(A, B, C, M, K, N);
            return;
        }
        B_T_cap = need;
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B_T[k * N + j] = B[j * K + k];
        }
    }

    for (int ii = 0; ii < M; ii += tile_size) {
        for (int jj = 0; jj < N; jj += tile_size) {
            int i_end = (ii + tile_size < M) ? ii + tile_size : M;
            int j_end = (jj + tile_size < N) ? jj + tile_size : N;
            for (int kk = 0; kk < K; kk += tile_size) {
                int k_end = (kk + tile_size < K) ? kk + tile_size : K;
                for (int i = ii; i < i_end; i++) {
                    int a_row = i * K;
                    int c_row = i * N;
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[a_row + k];
                        int bt_row = k * N;
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B_T[bt_row + j];
                        }
                    }
                }
            }
        }
    }
}

void mat_mat_mult_transposeA_rtile(float* A, float* B, float* C,
                                   int M, int K, int N, int tile_size) {
    if (tile_size <= 0)
        tile_size = BLOCK_SIZE;
    /* Reused across calls; this routine is not safe for concurrent callers. */
    static float *A_T = NULL;
    static size_t A_T_cap = 0;
    size_t need = (size_t)K * M * sizeof(float);

    if (need > A_T_cap) {
        free(A_T);
        A_T = NULL;
        A_T_cap = 0;
        if (posix_memalign((void**)&A_T, 64, need) != 0) {
            mat_mat_mult_transposeA_naive(A, B, C, M, K, N);
            return;
        }
        A_T_cap = need;
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A_T[k * M + i] = A[i * K + k];
        }
    }

    for (int ii = 0; ii < K; ii += tile_size) {
        for (int jj = 0; jj < N; jj += tile_size) {
            int i_end = (ii + tile_size < K) ? ii + tile_size : K;
            int j_end = (jj + tile_size < N) ? jj + tile_size : N;
            for (int kk = 0; kk < M; kk += tile_size) {
                int k_end = (kk + tile_size < M) ? kk + tile_size : M;
                for (int i = ii; i < i_end; i++) {
                    int at_row = i * M;
                    int c_row = i * N;
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A_T[at_row + k];
                        int b_row = k * N;
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B[b_row + j];
                        }
                    }
                }
            }
        }
    }
}
