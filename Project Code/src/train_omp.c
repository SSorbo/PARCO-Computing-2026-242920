#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "math_utils.h"
#include "math_omp.h"
#include "train.h"
#include "train_utils.h"

// Accumulate bias gradients for both layers inside a single parallel
// region.  Uses row-major traversal (cache-friendly) with per-thread
// partial sums to avoid concurrent updates to shared gradients. Reductions
// sum thread partials in thread-ID order, making repeated runs deterministic
// for a fixed team.
static void accumulate_both_bias_gradients_omp(
    float* o_deltas, float* o_bias_grads, int batch_count, int out_size,
    float* h_deltas, float* h_bias_grads, int hid_size)
{
    int max_threads = omp_get_max_threads();
    int max_cols = (hid_size > out_size) ? hid_size : out_size;

    // Reused across mini-batches; concurrent host calls to train_openmp would
    // require separate storage.
    static float* thread_buf = NULL;
    static size_t buf_cap = 0;
    size_t need = (size_t)max_threads * max_cols * sizeof(float);

    if (need > buf_cap) {
        free(thread_buf);
        thread_buf = NULL;
        buf_cap = 0;
        if (posix_memalign((void**)&thread_buf, 64, need) != 0) {
            accumulate_bias_gradients(o_deltas, o_bias_grads,
                                      batch_count, out_size);
            accumulate_bias_gradients(h_deltas, h_bias_grads,
                                      batch_count, hid_size);
            return;
        }
        buf_cap = need;
    }
    // Zero per-thread region; output accumulation below relies on it.
    memset(thread_buf, 0, need);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        float* my = thread_buf + tid * max_cols;

        // --- Output layer ---
        #pragma omp for schedule(static)
        for (int sample = 0; sample < batch_count; sample++) {
            int base = sample * out_size;
            for (int col = 0; col < out_size; col++)
                my[col] += o_deltas[base + col];
        }

        #pragma omp for schedule(static)
        for (int col = 0; col < out_size; col++) {
            float sum = 0.0f;
            for (int t = 0; t < max_threads; t++)
                sum += thread_buf[t * max_cols + col];
            o_bias_grads[col] = sum;
        }

        // --- Hidden layer ---
        for (int c = 0; c < hid_size; c++) my[c] = 0.0f;

        #pragma omp for schedule(static)
        for (int sample = 0; sample < batch_count; sample++) {
            int base = sample * hid_size;
            for (int col = 0; col < hid_size; col++)
                my[col] += h_deltas[base + col];
        }

        #pragma omp for schedule(static)
        for (int col = 0; col < hid_size; col++) {
            float sum = 0.0f;
            for (int t = 0; t < max_threads; t++)
                sum += thread_buf[t * max_cols + col];
            h_bias_grads[col] = sum;
        }
    }
}

static void zero_grads_omp(NeuralNetwork* nn) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    int w_hid_size = hid->input_size * hid->output_size;
    int w_out_size = out->input_size * out->output_size;

    // Bias arrays are much smaller than weight matrices, so one thread clears
    // them while the weight matrices are partitioned across the team.
    #pragma omp parallel
    {
        #pragma omp single
        {
            memset(hid->bias_grads, 0, hid->output_size * sizeof(float));
            memset(out->bias_grads, 0, out->output_size * sizeof(float));
        }

        #pragma omp for simd schedule(static)
        for (int i = 0; i < w_hid_size; i++)
            hid->weight_grads[i] = 0.0f;

        #pragma omp for simd schedule(static)
        for (int i = 0; i < w_out_size; i++)
            out->weight_grads[i] = 0.0f;
    }
}

static void update_weights_omp(NeuralNetwork* nn, float learning_rate) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    int w_hid_size = hid->input_size * hid->output_size;
    int w_out_size = out->input_size * out->output_size;

    // Bias arrays are updated by one thread inside the parallel region below;
    // the larger weight matrices are partitioned across the team.
    #pragma omp parallel
    {
        #pragma omp for simd schedule(static)
        for (int i = 0; i < w_hid_size; i++)
            hid->weights[i] -= learning_rate * hid->weight_grads[i];

        #pragma omp for simd schedule(static)
        for (int i = 0; i < w_out_size; i++)
            out->weights[i] -= learning_rate * out->weight_grads[i];

        #pragma omp single
        {
            for (int i = 0; i < hid->output_size; i++)
                hid->biases[i] -= learning_rate * hid->bias_grads[i];
            for (int i = 0; i < out->output_size; i++)
                out->biases[i]  -= learning_rate * out->bias_grads[i];
        }
    }
}

static void run_parallel_mini_batch(NeuralNetwork* nn,
                                    float** inputs,
                                    float** targets,
                                    int* batch_indices,
                                    int batch_count,
                                    float* batch_inputs,
                                    float* batch_targets,
                                    float* batch_h_pre,
                                    float* batch_h_act,
                                    float* batch_o_pre,
                                    float* batch_o_act,
                                    float* batch_h_deltas,
                                    float* batch_o_deltas) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    int in_size = hid->input_size;
    int hid_size = hid->output_size;
    int out_size = out->output_size;

    size_t hid_batch_size = (size_t)batch_count * hid_size;
    size_t out_batch_size = (size_t)batch_count * out_size;

    // Single parallel region for packing both inputs and targets.
    #pragma omp parallel
    {
        // Pack inputs and targets into contiguous buffers for batched GEMM.
        #pragma omp for schedule(static)
        for (int sample = 0; sample < batch_count; sample++) {
            memcpy(batch_inputs + (size_t)sample * in_size,
                   inputs[batch_indices[sample]],
                   in_size * sizeof(float));
        }

        #pragma omp for schedule(static)
        for (int sample = 0; sample < batch_count; sample++) {
            memcpy(batch_targets + (size_t)sample * out_size,
                   targets[batch_indices[sample]],
                   out_size * sizeof(float));
        }
    }

    memset(batch_h_pre, 0, hid_batch_size * sizeof(float));

    mat_mat_mult_tiled_transposeB(batch_inputs, hid->weights, batch_h_pre,
                      batch_count, in_size, hid_size);

    #pragma omp parallel for schedule(static)
    for (int sample = 0; sample < batch_count; sample++) {
        int base = sample * hid_size;
        for (int col = 0; col < hid_size; col++) {
            float pre = batch_h_pre[base + col] + hid->biases[col];
            batch_h_pre[base + col] = pre;
            batch_h_act[base + col] = sigmoid(pre);
        }
    }

    memset(batch_o_pre, 0, out_batch_size * sizeof(float));

    mat_mat_mult_tiled_transposeB(batch_h_act, out->weights, batch_o_pre,
                      batch_count, hid_size, out_size);

    // Fuse output bias-add + sigmoid + delta computation per sample.
    // Each sample is independent so order within the sample is preserved.
    #pragma omp parallel for schedule(static)
    for (int sample = 0; sample < batch_count; sample++) {
        int base = sample * out_size;
        for (int col = 0; col < out_size; col++) {
            float pre = batch_o_pre[base + col] + out->biases[col];
            batch_o_pre[base + col] = pre;
            batch_o_act[base + col] = sigmoid(pre);
        }
        for (int col = 0; col < out_size; col++) {
            float error = batch_o_act[base + col] - batch_targets[base + col];
            batch_o_deltas[base + col] = error * sigmoid_deriv(batch_o_pre[base + col]);
        }
    }

    memset(batch_h_deltas, 0, hid_batch_size * sizeof(float));

    mat_mat_mult_tiled(batch_o_deltas, out->weights, batch_h_deltas,
               batch_count, out_size, hid_size);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)hid_batch_size; i++)
        batch_h_deltas[i] *= sigmoid_deriv(batch_h_pre[i]);

    mat_mat_mult_tiled_transposeA(batch_o_deltas, batch_h_act, out->weight_grads,
                      batch_count, out_size, hid_size);

    mat_mat_mult_tiled_transposeA(batch_h_deltas, batch_inputs, hid->weight_grads,
                      batch_count, hid_size, in_size);

    accumulate_both_bias_gradients_omp(batch_o_deltas, out->bias_grads, batch_count, out_size,
                                       batch_h_deltas, hid->bias_grads, hid_size);
}

double train_openmp(NeuralNetwork* nn,
                       float** inputs,
                       float** targets,
                       int num_samples,
                       int epochs,
                       int batch_size,
                       int report_loss_each_epoch) {
    if (!nn || !inputs || !targets || num_samples <= 0 || epochs <= 0 || batch_size <= 0)
        return -1.0;

    if (report_loss_each_epoch)
        fprintf(stderr, "Training with OpenMP using batched GEMM...\n");

    int* batch_indices = (int*)malloc(batch_size * sizeof(int));
    if (!batch_indices)
        return -1.0;

    float* batch_inputs = NULL;
    float* batch_targets = NULL;
    float* batch_h_pre = NULL;
    float* batch_h_act = NULL;
    float* batch_o_pre = NULL;
    float* batch_o_act = NULL;
    float* batch_h_deltas = NULL;
    float* batch_o_deltas = NULL;

    int input_size = nn->hidden.input_size;
    int hid_size = nn->hidden.output_size;
    int out_size = nn->output.output_size;

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
        free(batch_inputs);
        free(batch_targets);
        free(batch_h_pre);
        free(batch_h_act);
        free(batch_o_pre);
        free(batch_o_act);
        free(batch_h_deltas);
        free(batch_o_deltas);
        free(batch_indices);
        return -1.0;
    }

    const float learning_rate = 0.5f;
    double train_time = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = omp_get_wtime();
        for (int start = 0; start < num_samples; start += batch_size) {
            int batch_end = (start + batch_size < num_samples) ? start + batch_size : num_samples;
            int batch_count = batch_end - start;

            for (int j = 0; j < batch_count; j++)
                batch_indices[j] = start + j;

            zero_grads_omp(nn);
            run_parallel_mini_batch(nn,
                                    inputs,
                                    targets,
                                    batch_indices,
                                    batch_count,
                                    batch_inputs,
                                    batch_targets,
                                    batch_h_pre,
                                    batch_h_act,
                                    batch_o_pre,
                                    batch_o_act,
                                    batch_h_deltas,
                                    batch_o_deltas);

            float lr_per_sample = learning_rate / (float)batch_count;
            update_weights_omp(nn, lr_per_sample);
        }
        train_time += omp_get_wtime() - t0;

        if (report_loss_each_epoch) {
            float loss = compute_training_loss(
                nn, inputs, targets, num_samples,
                nn->hidden.input_size, nn->hidden.output_size,
                nn->output.output_size);
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
    free(batch_indices);

    return train_time;
}
