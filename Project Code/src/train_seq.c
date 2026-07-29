#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "math_utils.h"
#include "train.h"
#include "train_utils.h"

/* Monotonic wall-clock time without an OpenMP dependency. */
static double get_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double train_sequential(NeuralNetwork* nn,
                           float** inputs,
                           float** targets,
                           int num_samples,
                           int epochs,
                           int batch_size,
                           int report_loss_each_epoch) {
    if (!nn || !inputs || !targets || num_samples <= 0 || epochs <= 0 || batch_size <= 0)
        return -1.0;

    int* indices = (int*)malloc((size_t)batch_size * sizeof(int));
    if (!indices)
        return -1.0;

    int input_size = nn->hidden.input_size;
    int hid_size = nn->hidden.output_size;
    int out_size = nn->output.output_size;

    float* batch_inputs = NULL;
    float* batch_targets = NULL;
    float* batch_h_pre = NULL;
    float* batch_h_act = NULL;
    float* batch_o_pre = NULL;
    float* batch_o_act = NULL;
    float* batch_h_deltas = NULL;
    float* batch_o_deltas = NULL;

    if (posix_memalign((void**)&batch_inputs, 64,
                       (size_t)batch_size * input_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_targets, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_h_pre, 64,
                       (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_h_act, 64,
                       (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_o_pre, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_o_act, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_h_deltas, 64,
                       (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_o_deltas, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0) {
        free(indices);
        free(batch_inputs);
        free(batch_targets);
        free(batch_h_pre);
        free(batch_h_act);
        free(batch_o_pre);
        free(batch_o_act);
        free(batch_h_deltas);
        free(batch_o_deltas);
        return -1.0;
    }

    const float learning_rate = 0.5f;
    double train_time = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = get_wtime();
        for (int start = 0; start < num_samples; start += batch_size) {
            int batch_end = (start + batch_size < num_samples) ? start + batch_size : num_samples;
            int batch_count = batch_end - start;
            float lr_per_sample = learning_rate / (float)batch_count;

            for (int j = 0; j < batch_count; j++)
                indices[j] = start + j;

            zero_grads(nn);

            pack_batch(batch_inputs, inputs, indices, batch_count, input_size);
            pack_batch(batch_targets, targets, indices, batch_count, out_size);

            int hid_batch_size = batch_count * hid_size;
            int out_batch_size = batch_count * out_size;

            memset(batch_h_pre, 0, hid_batch_size * sizeof(float));
            mat_mat_mult_transposeB(batch_inputs, nn->hidden.weights, batch_h_pre,
                                     batch_count, input_size, hid_size);
            add_bias_rowwise(batch_h_pre, nn->hidden.biases, batch_count, hid_size);
            sigmoid_batch_to(batch_h_pre, batch_h_act, hid_batch_size);

            memset(batch_o_pre, 0, out_batch_size * sizeof(float));
            mat_mat_mult_transposeB(batch_h_act, nn->output.weights, batch_o_pre,
                                     batch_count, hid_size, out_size);
            add_bias_rowwise(batch_o_pre, nn->output.biases, batch_count, out_size);
            sigmoid_batch_to(batch_o_pre, batch_o_act, out_batch_size);

            compute_output_deltas(batch_o_act, batch_targets, batch_o_pre,
                                  batch_o_deltas, out_batch_size);

            memset(batch_h_deltas, 0, hid_batch_size * sizeof(float));
            mat_mat_mult(batch_o_deltas, nn->output.weights, batch_h_deltas,
                         batch_count, out_size, hid_size);
            apply_sigmoid_deriv_inplace(batch_h_pre, batch_h_deltas, hid_batch_size);

            mat_mat_mult_transposeA(batch_o_deltas, batch_h_act, nn->output.weight_grads,
                                     batch_count, out_size, hid_size);
            accumulate_bias_gradients(batch_o_deltas, nn->output.bias_grads,
                                      batch_count, out_size);

            mat_mat_mult_transposeA(batch_h_deltas, batch_inputs, nn->hidden.weight_grads,
                                     batch_count, hid_size, input_size);
            accumulate_bias_gradients(batch_h_deltas, nn->hidden.bias_grads,
                                      batch_count, hid_size);

            update_weights(nn, lr_per_sample);
        }
        train_time += get_wtime() - t0;

        if (report_loss_each_epoch) {
            float loss = compute_training_loss(nn, inputs, targets, num_samples,
                                               input_size, hid_size, out_size);
            if (loss < 0.0f) {
                train_time = -1.0;
                break;
            }
            fprintf(stderr, "[Epoch %d/%d] training loss = %.6f\n",
                    epoch + 1, epochs, loss);
        }
    }

    free(batch_inputs);
    free(batch_targets);
    free(batch_h_pre);
    free(batch_h_act);
    free(batch_o_pre);
    free(batch_o_act);
    free(batch_h_deltas);
    free(batch_o_deltas);
    free(indices);

    return train_time;
}

double train_sequential_naive(NeuralNetwork* nn,
                                 float** inputs,
                                 float** targets,
                                 int num_samples,
                                 int epochs,
                                 int batch_size,
                                 int report_loss_each_epoch) {
    if (!nn || !inputs || !targets || num_samples <= 0 || epochs <= 0 || batch_size <= 0)
        return -1.0;

    int* indices = (int*)malloc((size_t)batch_size * sizeof(int));
    if (!indices)
        return -1.0;

    int input_size = nn->hidden.input_size;
    int hid_size = nn->hidden.output_size;
    int out_size = nn->output.output_size;

    float* batch_inputs = NULL;
    float* batch_targets = NULL;
    float* batch_h_pre = NULL;
    float* batch_h_act = NULL;
    float* batch_o_pre = NULL;
    float* batch_o_act = NULL;
    float* batch_h_deltas = NULL;
    float* batch_o_deltas = NULL;

    if (posix_memalign((void**)&batch_inputs, 64,
                       (size_t)batch_size * input_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_targets, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_h_pre, 64,
                       (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_h_act, 64,
                       (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_o_pre, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_o_act, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_h_deltas, 64,
                       (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&batch_o_deltas, 64,
                       (size_t)batch_size * out_size * sizeof(float)) != 0) {
        free(indices);
        free(batch_inputs);
        free(batch_targets);
        free(batch_h_pre);
        free(batch_h_act);
        free(batch_o_pre);
        free(batch_o_act);
        free(batch_h_deltas);
        free(batch_o_deltas);
        return -1.0;
    }

    const float learning_rate = 0.5f;
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    double train_time = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = get_wtime();
        for (int start = 0; start < num_samples; start += batch_size) {
            int batch_end = (start + batch_size < num_samples) ? start + batch_size : num_samples;
            int batch_count = batch_end - start;
            float lr_per_sample = learning_rate / (float)batch_count;

            for (int j = 0; j < batch_count; j++)
                indices[j] = start + j;

            zero_grads(nn);

            pack_batch(batch_inputs, inputs, indices, batch_count, input_size);
            pack_batch(batch_targets, targets, indices, batch_count, out_size);

            int hid_batch_size = batch_count * hid_size;
            int out_batch_size = batch_count * out_size;

            /* ----- forward pass (naive transposeB) ----- */
            memset(batch_h_pre, 0, hid_batch_size * sizeof(float));
            mat_mat_mult_naive(batch_inputs, hid->weights, batch_h_pre,
                               batch_count, input_size, hid_size);
            add_bias_rowwise(batch_h_pre, hid->biases, batch_count, hid_size);
            sigmoid_batch_to(batch_h_pre, batch_h_act, hid_batch_size);

            memset(batch_o_pre, 0, out_batch_size * sizeof(float));
            mat_mat_mult_naive(batch_h_act, out->weights, batch_o_pre,
                               batch_count, hid_size, out_size);
            add_bias_rowwise(batch_o_pre, out->biases, batch_count, out_size);
            sigmoid_batch_to(batch_o_pre, batch_o_act, out_batch_size);

            /* ----- backprop (naive loops for standard matmul and transposeA) ----- */
            compute_output_deltas(batch_o_act, batch_targets, batch_o_pre,
                                  batch_o_deltas, out_batch_size);

            /* Naive standard matmul: batch_h_deltas += o_deltas * output_weights */
            memset(batch_h_deltas, 0, hid_batch_size * sizeof(float));
            for (int i = 0; i < batch_count; i++) {
                for (int k = 0; k < out_size; k++) {
                    float a = batch_o_deltas[i * out_size + k];
                    for (int j = 0; j < hid_size; j++) {
                        batch_h_deltas[i * hid_size + j] +=
                            a * out->weights[k * hid_size + j];
                    }
                }
            }
            apply_sigmoid_deriv_inplace(batch_h_pre, batch_h_deltas, hid_batch_size);

            /* Naive transposeA: output_weight_grads += o_deltas^T * h_act */
            for (int k = 0; k < out_size; k++) {
                for (int i = 0; i < batch_count; i++) {
                    float a = batch_o_deltas[i * out_size + k];
                    for (int j = 0; j < hid_size; j++) {
                        out->weight_grads[k * hid_size + j] +=
                            a * batch_h_act[i * hid_size + j];
                    }
                }
            }
            accumulate_bias_gradients(batch_o_deltas, out->bias_grads,
                                      batch_count, out_size);

            /* Naive transposeA: hidden_weight_grads += h_deltas^T * inputs */
            for (int k = 0; k < hid_size; k++) {
                for (int i = 0; i < batch_count; i++) {
                    float a = batch_h_deltas[i * hid_size + k];
                    for (int j = 0; j < input_size; j++) {
                        hid->weight_grads[k * input_size + j] +=
                            a * batch_inputs[i * input_size + j];
                    }
                }
            }
            accumulate_bias_gradients(batch_h_deltas, hid->bias_grads,
                                      batch_count, hid_size);

            update_weights(nn, lr_per_sample);
        }
        train_time += get_wtime() - t0;

        if (report_loss_each_epoch) {
            float loss = compute_training_loss(nn, inputs, targets, num_samples,
                                               input_size, hid_size, out_size);
            if (loss < 0.0f) {
                train_time = -1.0;
                break;
            }
            fprintf(stderr, "[Epoch %d/%d] training loss = %.6f\n",
                    epoch + 1, epochs, loss);
        }
    }

    free(batch_inputs);
    free(batch_targets);
    free(batch_h_pre);
    free(batch_h_act);
    free(batch_o_pre);
    free(batch_o_act);
    free(batch_h_deltas);
    free(batch_o_deltas);
    free(indices);

    return train_time;
}
