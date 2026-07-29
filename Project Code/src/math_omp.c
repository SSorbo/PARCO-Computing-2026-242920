#include "math_omp.h"
#include "math_utils.h"
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

#define BLOCK_SIZE 64

// C += A * B, where A is MxK, B is KxN, and C is MxN.
// M is the number of rows in A and C.
// K is the shared (reduction) dimension.
// N is the number of columns in B and C.
void mat_mat_mult_tiled(float* A, float* B, float* C, int M, int K, int N) {
    A = __builtin_assume_aligned(A, 64);
    B = __builtin_assume_aligned(B, 64);
    C = __builtin_assume_aligned(C, 64);

    #pragma omp parallel for collapse(2) schedule(static)
    // ii is the starting row index of the current block in matrix A and C
    // jj is the starting column index of the current block in matrix B and C
    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            // Compute the end indices for the current block, ensuring we don't exceed matrix dimensions.
            int i_end = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
            int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;

            // kk is the starting index for the current block in the shared dimension K
            for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                // Compute the end index for the current block in the K dimension.
                int k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;

                // Perform the multiplication for the current block.
                for (int i = ii; i < i_end; i++) {
                    // Calculate the starting index for the current row in matrix A and C
                    int a_row = i * K;
                    int c_row = i * N;

                    // Perform the multiplication for the current block in the K dimension.
                    for (int k = kk; k < k_end; k++) {
                        float a_val = A[a_row + k];
                        int b_row = k * N;

                        #pragma omp simd aligned(B, C: 64)
                        for (int j = jj; j < j_end; j++) {
                            C[c_row + j] += a_val * B[b_row + j];
                        }
                    }
                }
            }
        }
    }
}

// C += A * B^T, where A is MxK, B is NxK, and C is MxN.
// B is supplied in output-major form (one K-element row per output).
// Packing B^T as a KxN matrix makes the inner j loop contiguous in B_T and C.
void mat_mat_mult_tiled_transposeB(float* A, float* B, float* C,
                                   int M, int K, int N) {
    // Reuse the aligned packing buffer across calls. The kernel is therefore
    // not re-entrant: separate host threads must not call it concurrently.
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

    A = __builtin_assume_aligned(A, 64);
    C = __builtin_assume_aligned(C, 64);
    B_T = __builtin_assume_aligned(B_T, 64);

    #pragma omp parallel
    {
        // Pack B from NxK to KxN. Each (k,j) destination is written once.
        #pragma omp for collapse(2) schedule(static)
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                B_T[k * N + j] = B[j * K + k];
            }
        }

        // The implicit barrier above guarantees that packing is complete.
        // Each (ii,jj) pair owns one output tile for every kk block, so no
        // two threads update the same element of C and no atomics are needed.
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                // Clamp the final output tile at the matrix edges.
                int i_end = (ii + BLOCK_SIZE < M) ? ii + BLOCK_SIZE : M;
                int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;

                // kk advances through blocks of the shared K dimension.
                for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
                    int k_end = (kk + BLOCK_SIZE < K) ? kk + BLOCK_SIZE : K;

                    for (int i = ii; i < i_end; i++) {
                        // Row offsets avoid repeated row-index multiplication.
                        int a_row = i * K;
                        int c_row = i * N;

                        for (int k = kk; k < k_end; k++) {
                            float a_val = A[a_row + k];
                            int bt_row = k * N;

                            // j walks contiguous rows of packed B_T and C.
                            #pragma omp simd aligned(B_T, C: 64)
                            for (int j = jj; j < j_end; j++) {
                                C[c_row + j] += a_val * B_T[bt_row + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

// C += A^T * B, where A is MxK, B is MxN, and C is KxN.
// Packing A^T as a KxM matrix turns each reduction row into contiguous data.
// The output tile grid is KxN; the shared (reduction) dimension is M.
void mat_mat_mult_tiled_transposeA(float* A, float* B, float* C,
                                   int M, int K, int N) {
    // Reuse the aligned packing buffer across calls. The kernel is therefore
    // not re-entrant: separate host threads must not call it concurrently.
    static float *A_T = NULL;
    static size_t A_T_cap = 0;
    size_t need = (size_t)K * M * sizeof(float);

    if (need > A_T_cap) {
        free(A_T);
        A_T = NULL;
        A_T_cap = 0;
        if (posix_memalign((void**)&A_T, 64, need) != 0) {
            mat_mat_mult_transposeA(A, B, C, M, K, N);
            return;
        }
        A_T_cap = need;
    }

    B = __builtin_assume_aligned(B, 64);
    C = __builtin_assume_aligned(C, 64);
    A_T = __builtin_assume_aligned(A_T, 64);

    #pragma omp parallel
    {
        // Pack A from MxK to KxM. Each (k,i) destination is written once.
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < M; i++) {
            for (int k = 0; k < K; k++) {
                A_T[k * M + i] = A[i * K + k];
            }
        }

        // The implicit barrier above guarantees that packing is complete.
        // Each (ii,jj) pair owns one KxN output tile for every kk block, so
        // no two threads update the same element of C.
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < K; ii += BLOCK_SIZE) {
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                // Clamp the final output tile at the matrix edges.
                int i_end = (ii + BLOCK_SIZE < K) ? ii + BLOCK_SIZE : K;
                int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;

                // kk advances through blocks of the shared M dimension.
                for (int kk = 0; kk < M; kk += BLOCK_SIZE) {
                    int k_end = (kk + BLOCK_SIZE < M) ? kk + BLOCK_SIZE : M;

                    for (int i = ii; i < i_end; i++) {
                        // Row offsets avoid repeated row-index multiplication.
                        int at_row = i * M;
                        int c_row = i * N;

                        for (int k = kk; k < k_end; k++) {
                            float a_val = A_T[at_row + k];
                            int b_row = k * N;

                            // j walks contiguous rows of B and C.
                            #pragma omp simd aligned(B, C: 64)
                            for (int j = jj; j < j_end; j++) {
                                C[c_row + j] += a_val * B[b_row + j];
                            }
                        }
                    }
                }
            }
        }
    }
}
