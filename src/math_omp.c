#include "math_omp.h"
#include <omp.h>
#include <stddef.h>

#define BLOCK_SIZE 64

void mat_vec_mult_tiled(float* weights, float* input_vec, float* bias,
                        float* output, int rows, int cols) {
    /*
     * Assert 64-byte alignment so the compiler can emit aligned-load
     * SIMD instructions (MOVAPS / VMOVAPS) instead of unaligned variants.
     * __builtin_assume_aligned is supported by both GCC and Clang; on
     * compilers that don't support it the macro is a no-op.
     */
    weights   = __builtin_assume_aligned(weights,   64);
    input_vec = __builtin_assume_aligned(input_vec, 64);
    output    = __builtin_assume_aligned(output,    64);

    /* Initialise output with bias (or zero if no bias vector supplied). */
    if (bias) {
        for (int i = 0; i < rows; i++)
            output[i] = bias[i];
    } else {
        for (int i = 0; i < rows; i++)
            output[i] = 0.0f;
    }

    /*
     * =============================================================
     * CACHE-TILED MATRIX-VECTOR MULTIPLICATION  (4-deep loop nest)
     * =============================================================
     *
     * Naive mat-vec (rows × cols) streams the entire weight matrix
     * once but may evict x[] from L1/L2 cache before a row tile
     * finishes when rows and cols are large.  Blocking carves the
     * iteration space into BLOCK_SIZE×BLOCK_SIZE tiles so that:
     *
     *   - The tile of weights (BLOCK_SIZE rows × BLOCK_SIZE cols)
     *     stays in L2 cache while those rows are processed.
     *   - The corresponding slice of input_vec (BLOCK_SIZE elements)
     *     fits comfortably in L1 (64 × 4 B = 256 B, half a typical
     *     512-bit L1 cache line set).
     *
     * Loop structure:
     *   Layer 1 (ii)  — row    tile index   [outermost]
     *   Layer 2 (jj)  — column tile index
     *   Layer 3 (i)   — row    inside tile
     *   Layer 4 (j)   — column inside tile  [innermost — SIMD]
     */

    for (int ii = 0; ii < rows; ii += BLOCK_SIZE) {
        int i_end = (ii + BLOCK_SIZE < rows) ? ii + BLOCK_SIZE : rows;

        for (int jj = 0; jj < cols; jj += BLOCK_SIZE) {
            int j_end = (jj + BLOCK_SIZE < cols) ? jj + BLOCK_SIZE : cols;

            /*
             * Process every row inside the current row-tile.
             * Each row accumulates its partial dot-product over
             * the current column-tile into a scalar sum, which
             * is then folded into output[i].
             */
            for (int i = ii; i < i_end; i++) {
                float sum = 0.0f;
                int base = i * cols;

                /*
                 * INNERMOST LOOP — SIMD vectorised multiply-add.
                 *
                 * #pragma omp simd tells the compiler to ignore
                 * loop-carried dependencies on `sum` (relaxed with
                 * reduction) and emit packed SIMD instructions
                 * (AVX2 / ARM NEON) for the multiply-add.
                 *
                 * `aligned(weights, input_vec: 64)` asserts that
                 * weights[base] and input_vec[jj] are 64-byte
                 * aligned, allowing the compiler to use aligned
                 * loads (VMOVAPS) which have 1-cycle throughput
                 * vs. 2 cycles for unaligned on x86.
                 *
                 * Pre-condition: cols is a multiple of 16 floats
                 * (64 bytes / 4 B) so that each row starts aligned.
                 */
                #pragma omp simd aligned(weights, input_vec: 64) \
                                 reduction(+:sum)
                for (int j = jj; j < j_end; j++) {
                    sum += weights[base + j] * input_vec[j];
                }

                output[i] += sum;
            }
        }
    }
}
