#ifndef MATH_UTILS_H
#define MATH_UTILS_H

/* Batched matrix-matrix multiplication routines for sequential mini-batch processing.
 * All matrices are stored in row-major order.
 * The transpose variants reuse internal packing buffers and are not re-entrant.
 */

/* Naive O(M*K*N)
 * Serves as the performance floor for benchmarking.
 * Computes C += A * B^T  where A is MxK, B is NxK, C is MxN.
 */
void mat_mat_mult_naive(float* A, float* B, float* C,
                        int M, int K, int N);

/* Compute C += A * B where A is MxK and B is KxN. C is MxN. */
void mat_mat_mult(float* A, float* B, float* C,
                  int M, int K, int N);

float sigmoid(float x);
float sigmoid_deriv(float x);

/* Compute C += A * B^T where A is MxK and B is NxK. C is MxN. */
void mat_mat_mult_transposeB(float* A, float* B, float* C,
                             int M, int K, int N);

/* Compute C += A^T * B where A is MxK and B is MxN. C is KxN. */
void mat_mat_mult_transposeA(float* A, float* B, float* C,
                             int M, int K, int N);

/* ---- Runtime-tile-size variants (for tile-size sweep experiments) ---- */
void mat_mat_mult_rtile(float* A, float* B, float* C,
                        int M, int K, int N, int tile_size);

void mat_mat_mult_transposeB_rtile(float* A, float* B, float* C,
                                   int M, int K, int N, int tile_size);

void mat_mat_mult_transposeA_rtile(float* A, float* B, float* C,
                                   int M, int K, int N, int tile_size);

#endif /* MATH_UTILS_H */
