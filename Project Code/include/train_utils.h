#ifndef TRAIN_UTILS_H
#define TRAIN_UTILS_H

#include "nn.h"

/* Shared element-wise and utility helpers used by the training loops.
   Single-threaded; OpenMP variants live in train_omp.c. */

void pack_batch(float *batch_buf, float **source, int *batch_indices,
                int batch_count, int elem_count);

void add_bias_rowwise(float *matrix, float *bias, int batch_count, int cols);

void sigmoid_batch_to(const float *src, float *dst, int count);

void compute_output_deltas(float *output_activations, float *targets,
                           float *preacts, float *deltas, int count);

void apply_sigmoid_deriv_inplace(float *preacts, float *deltas, int count);

void accumulate_bias_gradients(float *deltas, float *bias_grads,
                               int batch_count, int cols);

void zero_grads(NeuralNetwork *nn);

void update_weights(NeuralNetwork *nn, float learning_rate);

/* Mean half-squared error per training sample. Returns -1 on error.
 * Reuses internal work buffers and is not re-entrant. */
float compute_training_loss(NeuralNetwork *nn, float **inputs, float **targets,
                            int num_samples, int input_size, int hid_size,
                            int out_size);

#endif
