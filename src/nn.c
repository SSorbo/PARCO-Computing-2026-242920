#include "nn.h"
#include "math_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size) {
    NeuralNetwork* nn = (NeuralNetwork*)aligned_alloc(64, sizeof(NeuralNetwork));
    if (!nn) return NULL;

    /* Zero the struct so unset pointers are NULL (safe for nn_free on early failure). */
    memset(nn, 0, sizeof(NeuralNetwork));

    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;

    hid->input_size  = input_size;
    hid->output_size = hidden_size;
    out->input_size  = hidden_size;
    out->output_size = output_size;

    int w_hid_size = input_size  * hidden_size;
    int w_out_size = hidden_size * output_size;

    /*
     * Allocate every float array on a 64-byte boundary via posix_memalign.
     * Aligned loads/stores let the compiler emit SIMD instructions (AVX/SSE)
     * and avoid cache-line splits that double L1 latency.
     */
    #define ALIGNED_ALLOC(ptr, count) \
        if (posix_memalign((void**)&(ptr), 64, (count) * sizeof(float)) != 0) { \
            nn_free(nn); return NULL; \
        }

    ALIGNED_ALLOC(hid->weights,       w_hid_size);
    ALIGNED_ALLOC(hid->weight_grads,  w_hid_size);
    ALIGNED_ALLOC(hid->biases,        hidden_size);
    ALIGNED_ALLOC(hid->bias_grads,    hidden_size);
    ALIGNED_ALLOC(hid->activations,   hidden_size);
    ALIGNED_ALLOC(hid->pre_acts,      hidden_size);
    ALIGNED_ALLOC(hid->deltas,        hidden_size);

    ALIGNED_ALLOC(out->weights,       w_out_size);
    ALIGNED_ALLOC(out->weight_grads,  w_out_size);
    ALIGNED_ALLOC(out->biases,        output_size);
    ALIGNED_ALLOC(out->bias_grads,    output_size);
    ALIGNED_ALLOC(out->activations,   output_size);
    ALIGNED_ALLOC(out->pre_acts,      output_size);
    ALIGNED_ALLOC(out->deltas,        output_size);

    #undef ALIGNED_ALLOC

    /*
     * Initialize weights uniformly in [-0.5, 0.5] to break symmetry while
     * keeping pre-activation values small so sigmoid operates in its
     * linear-ish region early in training (avoiding saturation).
     */
    for (int i = 0; i < w_hid_size; i++)
        hid->weights[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    for (int i = 0; i < w_out_size; i++)
        out->weights[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;

    /* Biases start at zero — no reason to prefer any neuron initially. */
    for (int i = 0; i < hidden_size; i++)
        hid->biases[i] = 0.0f;
    for (int i = 0; i < output_size; i++)
        out->biases[i] = 0.0f;

    /* Gradients start at zero. */
    memset(hid->weight_grads, 0, w_hid_size * sizeof(float));
    memset(hid->bias_grads,   0, hidden_size * sizeof(float));
    memset(out->weight_grads, 0, w_out_size * sizeof(float));
    memset(out->bias_grads,   0, output_size * sizeof(float));

    return nn;
}

void nn_free(NeuralNetwork* nn) {
    if (!nn) return;
    Layer* layers[2] = { &nn->hidden, &nn->output };
    for (int layer_idx = 0; layer_idx < 2; layer_idx++) {
        free(layers[layer_idx]->weights);
        free(layers[layer_idx]->weight_grads);
        free(layers[layer_idx]->biases);
        free(layers[layer_idx]->bias_grads);
        free(layers[layer_idx]->activations);
        free(layers[layer_idx]->pre_acts);
        free(layers[layer_idx]->deltas);
    }
    free(nn);
}

NeuralNetwork* nn_clone(NeuralNetwork* src) {
    if (!src) return NULL;

    NeuralNetwork* clone = nn_create(src->hidden.input_size,
                                     src->hidden.output_size,
                                     src->output.output_size);
    if (!clone) return NULL;

    int hid_w_size = src->hidden.input_size * src->hidden.output_size;
    int out_w_size = src->output.input_size * src->output.output_size;

    memcpy(clone->hidden.weights, src->hidden.weights,
           hid_w_size * sizeof(float));
    memcpy(clone->hidden.biases, src->hidden.biases,
           src->hidden.output_size * sizeof(float));
    memcpy(clone->output.weights, src->output.weights,
           out_w_size * sizeof(float));
    memcpy(clone->output.biases, src->output.biases,
           src->output.output_size * sizeof(float));

    return clone;
}

void verify_identical_weights(NeuralNetwork* seq_nn, NeuralNetwork* omp_nn) {
    if (!seq_nn || !omp_nn) {
        fprintf(stderr, "ERROR: null neural network pointer passed to verification.\n");
        return;
    }

    int hid_w_size = seq_nn->hidden.input_size * seq_nn->hidden.output_size;
    int out_w_size = seq_nn->output.input_size * seq_nn->output.output_size;

    for (int i = 0; i < hid_w_size; i++) {
        float diff = fabsf(seq_nn->hidden.weights[i] - omp_nn->hidden.weights[i]);
        if (diff > 1e-4f) {
            fprintf(stderr,
                    "ERROR: Hidden weight divergence at index %d: seq=%.8f omp=%.8f diff=%.8f\n",
                    i, seq_nn->hidden.weights[i], omp_nn->hidden.weights[i], diff);
            return;
        }
    }
    for (int i = 0; i < seq_nn->hidden.output_size; i++) {
        float diff = fabsf(seq_nn->hidden.biases[i] - omp_nn->hidden.biases[i]);
        if (diff > 1e-4f) {
            fprintf(stderr,
                    "ERROR: Hidden bias divergence at index %d: seq=%.8f omp=%.8f diff=%.8f\n",
                    i, seq_nn->hidden.biases[i], omp_nn->hidden.biases[i], diff);
            return;
        }
    }
    for (int i = 0; i < out_w_size; i++) {
        float diff = fabsf(seq_nn->output.weights[i] - omp_nn->output.weights[i]);
        if (diff > 1e-4f) {
            fprintf(stderr,
                    "ERROR: Output weight divergence at index %d: seq=%.8f omp=%.8f diff=%.8f\n",
                    i, seq_nn->output.weights[i], omp_nn->output.weights[i], diff);
            return;
        }
    }
    for (int i = 0; i < seq_nn->output.output_size; i++) {
        float diff = fabsf(seq_nn->output.biases[i] - omp_nn->output.biases[i]);
        if (diff > 1e-4f) {
            fprintf(stderr,
                    "ERROR: Output bias divergence at index %d: seq=%.8f omp=%.8f diff=%.8f\n",
                    i, seq_nn->output.biases[i], omp_nn->output.biases[i], diff);
            return;
        }
    }

    printf("[VERIFICATION SUCCESS]: Both models converged to the exact same state.\n");
}

void nn_zero_grads(NeuralNetwork* nn) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    int w_hid_size = hid->input_size * hid->output_size;
    int w_out_size = out->input_size * out->output_size;
    memset(hid->weight_grads, 0, w_hid_size * sizeof(float));
    memset(hid->bias_grads,   0, hid->output_size * sizeof(float));
    memset(out->weight_grads, 0, w_out_size * sizeof(float));
    memset(out->bias_grads,   0, out->output_size * sizeof(float));
}

void nn_forward(NeuralNetwork* nn, float* input) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;

    int in_size  = hid->input_size;
    int hid_size = hid->output_size;
    int out_size = out->output_size;

    /*
     * Hidden layer: pre_acts[j] = sum_i(input[i] * W_hid[i][j]) + bias[j].
     *
     * W_hid is stored as [in_size x hid_size] row-major:
     * element (i,j) = weights[i * hid_size + j].
     * Outer loop over inputs, inner over outputs so that the inner loop
     * reads weights contiguously (i*hid_size+0, i*hid_size+1, ...) —
     * one cache line services multiple outputs before moving to the next input.
     */
    for (int j = 0; j < hid_size; j++)
        hid->pre_acts[j] = hid->biases[j];
    for (int i = 0; i < in_size; i++) {
        float in_val = input[i];
        int base = i * hid_size;
        for (int j = 0; j < hid_size; j++)
            hid->pre_acts[j] += in_val * hid->weights[base + j];
    }
    for (int j = 0; j < hid_size; j++)
        hid->activations[j] = sigmoid(hid->pre_acts[j]);

    /*
     * Output layer: pre_acts[j] = sum_i(A_hid[i] * W_out[i][j]) + bias[j].
     *
     * W_out is stored as [hid_size x out_size] row-major:
     * element (i,j) = weights[i * out_size + j].
     * Same cache-friendly access pattern: outer over inputs (hidden neurons),
     * inner over outputs — contiguous reads of the weight array.
     */
    for (int j = 0; j < out_size; j++)
        out->pre_acts[j] = out->biases[j];
    for (int i = 0; i < hid_size; i++) {
        float hid_act = hid->activations[i];
        int base = i * out_size;
        for (int j = 0; j < out_size; j++)
            out->pre_acts[j] += hid_act * out->weights[base + j];
    }
    for (int j = 0; j < out_size; j++)
        out->activations[j] = sigmoid(out->pre_acts[j]);
}

void nn_backward(NeuralNetwork* nn, float* input, float* target) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;

    int hid_size = hid->output_size;
    int out_size = out->output_size;
    int in_size  = hid->input_size;

    /*
     * --- Output layer deltas ---
     * Loss = MSE: L = (a - t)^2  (factor of 2 absorbed by learning rate).
     * dL/dz_out = (a_out - t) * sigmoid'(z_out).
     */
    for (int j = 0; j < out_size; j++) {
        float error = out->activations[j] - target[j];
        out->deltas[j] = error * sigmoid_deriv(out->pre_acts[j]);
    }

    /*
     * --- Hidden layer deltas ---
     * Chain rule: dL/da_hid[i] = sum_j( dL/dz_out[j] * dz_out[j]/da_hid[i] )
     *                          = sum_j( delta_out[j] * W_out[i][j] )
     *
     * W_out is [hid_size x out_size] row-major:
     * element (i,j) is at W_out[i * out_size + j].
     *
     * CRITICAL: We do NOT physically transpose the weight matrix for
     * backprop. Instead we loop so the outer index walks hidden neurons
     * (i) and the inner loops over output neurons (j), reading
     * W_out[i * out_size + j] contiguously across each row. This is
     * mathematically equivalent to multiplying the stored matrix by
     * delta_out — no explicit transpose buffer needed.
     */
    for (int i = 0; i < hid_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < out_size; j++) {
            sum += out->weights[i * out_size + j] * out->deltas[j];
        }
        hid->deltas[i] = sum * sigmoid_deriv(hid->pre_acts[i]);
    }

    /*
     * --- Accumulate output layer gradients ---
     * dW_out[i][j] += A_hid[i] * delta_out[j]
     * db_out[j]    += delta_out[j]
     *
     * Outer loop over hidden (rows), inner over output (cols) to match
     * the row-major layout of weight_grads — contiguous writes per row.
     */
    for (int i = 0; i < hid_size; i++) {
        float hid_act = hid->activations[i];
        int base = i * out_size;
        for (int j = 0; j < out_size; j++) {
            out->weight_grads[base + j] += hid_act * out->deltas[j];
        }
    }
    for (int j = 0; j < out_size; j++)
        out->bias_grads[j] += out->deltas[j];

    /*
     * --- Accumulate hidden layer gradients ---
     * dW_hid[i][j] += X[i] * delta_hid[j]
     * db_hid[j]    += delta_hid[j]
     */
    for (int i = 0; i < in_size; i++) {
        float in_val = input[i];
        int base = i * hid_size;
        for (int j = 0; j < hid_size; j++) {
            hid->weight_grads[base + j] += in_val * hid->deltas[j];
        }
    }
    for (int j = 0; j < hid_size; j++)
        hid->bias_grads[j] += hid->deltas[j];
}

void nn_update_weights(NeuralNetwork* nn, float learning_rate) {
    Layer* layers[2] = { &nn->hidden, &nn->output };
    for (int layer_idx = 0; layer_idx < 2; layer_idx++) {
        Layer* layer = layers[layer_idx];
        int weight_count = layer->input_size * layer->output_size;
        int bias_count   = layer->output_size;

        /*
         * SGD step: W -= lr * dW,  b -= lr * db.
         * Caller passes lr already divided by batch_size so these are
         * mean gradients over the mini-batch.
         */
        for (int i = 0; i < weight_count; i++)
            layer->weights[i] -= learning_rate * layer->weight_grads[i];
        for (int i = 0; i < bias_count; i++)
            layer->biases[i]  -= learning_rate * layer->bias_grads[i];
    }
}
