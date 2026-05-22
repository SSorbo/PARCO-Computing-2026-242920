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
