#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "nn.h"

static NeuralNetwork* create_thread_local_nn(NeuralNetwork* master) {
    if (!master) return NULL;

    NeuralNetwork* local_nn = NULL;
    if (posix_memalign((void**)&local_nn, 64, sizeof(NeuralNetwork)) != 0)
        return NULL;
    memset(local_nn, 0, sizeof(NeuralNetwork));

    Layer* hid = &local_nn->hidden;
    Layer* out = &local_nn->output;

    hid->input_size  = master->hidden.input_size;
    hid->output_size = master->hidden.output_size;
    out->input_size  = master->output.input_size;
    out->output_size = master->output.output_size;

    hid->weights = master->hidden.weights;
    hid->biases  = master->hidden.biases;
    out->weights = master->output.weights;
    out->biases  = master->output.biases;

    int hid_w_size = hid->input_size * hid->output_size;
    int out_w_size = out->input_size * out->output_size;

    if (posix_memalign((void**)&hid->weight_grads, 64, hid_w_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&hid->bias_grads, 64, hid->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&hid->activations, 64, hid->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&hid->pre_acts, 64, hid->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&hid->deltas, 64, hid->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&out->weight_grads, 64, out_w_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&out->bias_grads, 64, out->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&out->activations, 64, out->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&out->pre_acts, 64, out->output_size * sizeof(float)) != 0)
        goto fail;
    if (posix_memalign((void**)&out->deltas, 64, out->output_size * sizeof(float)) != 0)
        goto fail;

    memset(hid->weight_grads, 0, hid_w_size * sizeof(float));
    memset(hid->bias_grads,   0, hid->output_size * sizeof(float));
    memset(hid->activations,  0, hid->output_size * sizeof(float));
    memset(hid->pre_acts,     0, hid->output_size * sizeof(float));
    memset(hid->deltas,       0, hid->output_size * sizeof(float));
    memset(out->weight_grads, 0, out_w_size * sizeof(float));
    memset(out->bias_grads,   0, out->output_size * sizeof(float));
    memset(out->activations,  0, out->output_size * sizeof(float));
    memset(out->pre_acts,     0, out->output_size * sizeof(float));
    memset(out->deltas,       0, out->output_size * sizeof(float));

    return local_nn;

fail:
    if (local_nn) {
        free(hid->weight_grads);
        free(hid->bias_grads);
        free(hid->activations);
        free(hid->pre_acts);
        free(hid->deltas);
        free(out->weight_grads);
        free(out->bias_grads);
        free(out->activations);
        free(out->pre_acts);
        free(out->deltas);
        free(local_nn);
    }
    return NULL;
}

static void destroy_thread_local_nn(NeuralNetwork* local_nn) {
    if (!local_nn) return;
    Layer* hid = &local_nn->hidden;
    Layer* out = &local_nn->output;
    free(hid->weight_grads);
    free(hid->bias_grads);
    free(hid->activations);
    free(hid->pre_acts);
    free(hid->deltas);
    free(out->weight_grads);
    free(out->bias_grads);
    free(out->activations);
    free(out->pre_acts);
    free(out->deltas);
    free(local_nn);
}

static void reduce_thread_local_gradients(NeuralNetwork* nn,
                                          NeuralNetwork** thread_nns,
                                          int num_threads) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    int hid_w_size = hid->input_size * hid->output_size;
    int out_w_size = out->input_size * out->output_size;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < hid_w_size; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++)
            sum += thread_nns[t]->hidden.weight_grads[i];
        hid->weight_grads[i] = sum;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < hid->output_size; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++)
            sum += thread_nns[t]->hidden.bias_grads[i];
        hid->bias_grads[i] = sum;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out_w_size; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++)
            sum += thread_nns[t]->output.weight_grads[i];
        out->weight_grads[i] = sum;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < out->output_size; i++) {
        float sum = 0.0f;
        for (int t = 0; t < num_threads; t++)
            sum += thread_nns[t]->output.bias_grads[i];
        out->bias_grads[i] = sum;
    }
}

static void run_parallel_mini_batch(NeuralNetwork* nn,
                                    float** inputs,
                                    float** targets,
                                    int* batch_indices,
                                    int batch_size,
                                    NeuralNetwork** thread_nns,
                                    int num_threads) {
    nn_zero_grads(nn);

    #pragma omp parallel for schedule(static)
    for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        int tid = omp_get_thread_num();
        NeuralNetwork* local_nn = thread_nns[tid];
        int sample_idx = batch_indices[batch_idx];
        nn_forward(local_nn, inputs[sample_idx]);
        nn_backward(local_nn, inputs[sample_idx], targets[sample_idx]);
    }

    reduce_thread_local_gradients(nn, thread_nns, num_threads);
}

void train_openmp(NeuralNetwork* nn,
                  float** inputs,
                  float** targets,
                  int num_samples,
                  int epochs,
                  int batch_size) {
    if (!nn || !inputs || !targets || num_samples <= 0 || epochs <= 0 || batch_size <= 0)
        return;

    int num_threads = omp_get_max_threads();
    if (num_threads < 1)
        num_threads = 1;

    printf("Training with OpenMP using %d threads...\n", num_threads);

    NeuralNetwork** thread_nns = (NeuralNetwork**)malloc(num_threads * sizeof(NeuralNetwork*));
    if (!thread_nns)
        return;

    for (int t = 0; t < num_threads; t++) {
        thread_nns[t] = create_thread_local_nn(nn);
        if (!thread_nns[t]) {
            for (int j = 0; j < t; j++)
                destroy_thread_local_nn(thread_nns[j]);
            free(thread_nns);
            return;
        }
    }

    int* batch_indices = (int*)malloc(batch_size * sizeof(int));
    if (!batch_indices) {
        for (int t = 0; t < num_threads; t++)
            destroy_thread_local_nn(thread_nns[t]);
        free(thread_nns);
        return;
    }

    const float learning_rate = 0.5f;
    const float lr_per_sample = learning_rate / (float)batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int start = 0; start < num_samples; start += batch_size) {
            int batch_end = (start + batch_size < num_samples) ? start + batch_size : num_samples;
            int batch_count = batch_end - start;

            for (int j = 0; j < batch_count; j++)
                batch_indices[j] = start + j;

            nn_zero_grads(nn);
            for (int t = 0; t < num_threads; t++) {
                int hid_w_size = nn->hidden.input_size * nn->hidden.output_size;
                int out_w_size = nn->output.input_size * nn->output.output_size;
                memset(thread_nns[t]->hidden.weight_grads, 0, hid_w_size * sizeof(float));
                memset(thread_nns[t]->hidden.bias_grads, 0, nn->hidden.output_size * sizeof(float));
                memset(thread_nns[t]->output.weight_grads, 0, out_w_size * sizeof(float));
                memset(thread_nns[t]->output.bias_grads, 0, nn->output.output_size * sizeof(float));
            }

            run_parallel_mini_batch(nn, inputs, targets, batch_indices, batch_count, thread_nns, num_threads);
            nn_update_weights(nn, lr_per_sample);
        }
    }

    free(batch_indices);
    for (int t = 0; t < num_threads; t++)
        destroy_thread_local_nn(thread_nns[t]);
    free(thread_nns);
}
