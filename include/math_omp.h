#ifndef MATH_OMP_H
#define MATH_OMP_H

/*
 * Tiled matrix-vector multiplication:  output = weights * input_vec + bias.
 *
 * Uses cache blocking (BLOCK_SIZE x BLOCK_SIZE tiles) to keep a tile of the
 * weight matrix and the corresponding slice of the input vector in L1/L2
 * cache, avoiding capacity misses on large matrices.  The innermost loop
 * carries an OpenMP SIMD pragma so the compiler emits packed SIMD
 * instructions (AVX2 / NEON) for the multiply-add reduction.
 *
 * All float pointers must be 64-byte aligned (allocated via posix_memalign
 * or aligned_alloc) — the SIMD pragma asserts this so the compiler can
 * emit aligned loads.
 */
void mat_vec_mult_tiled(float* weights, float* input_vec, float* bias,
                        float* output, int rows, int cols);

#endif /* MATH_OMP_H */
