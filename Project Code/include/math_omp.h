#ifndef MATH_OMP_H
#define MATH_OMP_H

/*
 * Tiled matrix-matrix multiplication routines for batched neural network
 * mini-batch processing. All matrices are row-major.
 * The transpose variants reuse internal packing buffers and are not re-entrant.
 */
/* Compute C += A * B where A is MxK and B is KxN. C is MxN. */
void mat_mat_mult_tiled(float* A, float* B, float* C,
                        int M, int K, int N);

/* Compute C += A * B^T where A is MxK and B is NxK. C is MxN. */
void mat_mat_mult_tiled_transposeB(float* A, float* B, float* C,
                                   int M, int K, int N);

/* Compute C += A^T * B where A is MxK and B is MxN. C is KxN. */
void mat_mat_mult_tiled_transposeA(float* A, float* B, float* C,
                                   int M, int K, int N);

#endif /* MATH_OMP_H */
