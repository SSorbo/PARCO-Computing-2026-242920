#include "../include/math_utils.h"
#include <math.h>

void mat_vec_mult(float* weights, float* input_vec, float* bias, float* output,
                  int rows, int cols) {
    /*
     * `weights` is a row-major matrix with dimensions rows x cols.
     * Outer loop iterates rows, inner loop iterates cols to favor C row-major
     * layout: access element (i,j) as weights[i * cols + j]. This maximizes
     * L1 cache hits.
     */
    for (int i = 0; i < rows; ++i) {
        float sum = 0.0f;
        int base = i * cols;
        for (int j = 0; j < cols; ++j) {
            sum += weights[base + j] * input_vec[j];
        }
        if (bias) sum += bias[i];
        output[i] = sum;
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_deriv(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float relu_deriv(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}
