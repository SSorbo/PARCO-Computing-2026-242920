#ifndef NN_H
#define NN_H

#include <stddef.h>

typedef struct {
    int input_size;
    int output_size;
    /*
     * CRITICAL: All 2D matrices must be flattened into 1D arrays.
     * For example, `weights` is a flattened matrix with size
     * `input_size * output_size` stored in row-major order so that
     * element (i, j) is accessed as `weights[i * output_size + j]`.
     */
    float *weights;       /* size: input_size * output_size (row-major) */
    float *weight_grads;  /* same size as weights */

    /* biases are length output_size */
    float *biases;        /* size: output_size */
    float *bias_grads;    /* size: output_size */

    /* per-unit buffers of length output_size */
    float *activations;   /* activation output (after nonlinearity) */
    float *pre_acts;      /* linear pre-activation values */
    float *deltas;        /* error terms for backprop */
} Layer;

typedef struct {
    Layer hidden; /* exactly one hidden layer */
    Layer output; /* exactly one output layer */
} NeuralNetwork;

/* Lifecycle and training API */
NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size);
void nn_free(NeuralNetwork* nn);

/* Forward pass: computes activations for hidden and output layers.
 * `input` is an array of length `input_size`.
 */
void nn_forward(NeuralNetwork* nn, float* input);

/* Backward pass: computes gradients and fills weight/bias grads.
 * `input` is the original input; `target` is output-size vector.
 */
void nn_backward(NeuralNetwork* nn, float* input, float* target);

/* Update weights in-place using accumulated grads and given LR. */
void nn_update_weights(NeuralNetwork* nn, float learning_rate);

#endif /* NN_H */
