#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <stddef.h>

/* Multiply matrix `weights` (rows x cols) by vector `input_vec` (length cols),
 * add bias vector `bias` (length rows), store result in `output` (length rows).
 * Implementation must iterate outer loop over rows and inner over cols to
 * maximize cache locality for row-major storage: element weights[i * cols + j].
 */
void mat_vec_mult(float* weights, float* input_vec, float* bias, float* output,
                  int rows, int cols);

/* Activation functions and derivatives */
float sigmoid(float x);
float sigmoid_deriv(float x);
float relu(float x);
float relu_deriv(float x);

#endif /* MATH_UTILS_H */
