#include "train_utils.h"
#include "math_utils.h"
#include <stdlib.h>
#include <string.h>

void pack_batch(float *batch_buf, float **source, int *batch_indices,
                int batch_count, int elem_count) {
    for (int sample = 0; sample < batch_count; sample++) {
        memcpy(batch_buf + (size_t)sample * elem_count,
               source[batch_indices[sample]],
               elem_count * sizeof(float));
    }
}

void add_bias_rowwise(float *matrix, float *bias,
                      int batch_count, int cols) {
    for (int sample = 0; sample < batch_count; sample++) {
        for (int col = 0; col < cols; col++) {
            matrix[(size_t)sample * cols + col] += bias[col];
        }
    }
}

void sigmoid_batch_to(const float *src, float *dst, int count) {
    for (int i = 0; i < count; i++)
        dst[i] = sigmoid(src[i]);
}

void compute_output_deltas(float *output_activations,
                           float *targets,
                           float *preacts,
                           float *deltas,
                           int count) {
    for (int i = 0; i < count; i++) {
        float error = output_activations[i] - targets[i];
        deltas[i] = error * sigmoid_deriv(preacts[i]);
    }
}

void apply_sigmoid_deriv_inplace(float *preacts, float *deltas, int count) {
    for (int i = 0; i < count; i++)
        deltas[i] *= sigmoid_deriv(preacts[i]);
}

void accumulate_bias_gradients(float *deltas, float *bias_grads,
                               int batch_count, int cols) {
    // Row-major traversal: each sample's contiguous row is read once.
    // Caller must have zeroed bias_grads beforehand (zero_grads does this).
    for (int sample = 0; sample < batch_count; sample++) {
        int base = sample * cols;
        for (int col = 0; col < cols; col++)
            bias_grads[col] += deltas[base + col];
    }
}

void zero_grads(NeuralNetwork *nn) {
    Layer *hid = &nn->hidden;
    Layer *out = &nn->output;
    int w_hid_size = hid->input_size * hid->output_size;
    int w_out_size = out->input_size * out->output_size;
    memset(hid->weight_grads, 0, w_hid_size * sizeof(float));
    memset(hid->bias_grads,   0, hid->output_size * sizeof(float));
    memset(out->weight_grads, 0, w_out_size * sizeof(float));
    memset(out->bias_grads,   0, out->output_size * sizeof(float));
}

void update_weights(NeuralNetwork *nn, float learning_rate) {
    Layer *layers[2] = { &nn->hidden, &nn->output };
    for (int layer_idx = 0; layer_idx < 2; layer_idx++) {
        Layer *layer = layers[layer_idx];
        int weight_count = layer->input_size * layer->output_size;
        int bias_count   = layer->output_size;

        for (int i = 0; i < weight_count; i++)
            layer->weights[i] -= learning_rate * layer->weight_grads[i];
        for (int i = 0; i < bias_count; i++)
            layer->biases[i]  -= learning_rate * layer->bias_grads[i];
    }
}

float compute_training_loss(NeuralNetwork *nn, float **inputs, float **targets,
                            int num_samples, int input_size, int hid_size,
                            int out_size) {
    if (!nn || !inputs || !targets || num_samples <= 0 || input_size <= 0 ||
        hid_size <= 0 || out_size <= 0)
        return -1.0f;
    // Process samples in mini-batches so the GEMM sees M>1 rows and
    // can reuse the weight matrix across the batch.  Input and target
    // data are assumed to be stored contiguously (all dataset
    // constructors in this codebase use flat allocations with row
    // pointers), so we can address batches directly without copying.
    #define LOSS_BATCH 64

    int max_samples = (LOSS_BATCH < num_samples) ? LOSS_BATCH : num_samples;
    // Reused across evaluations; compute_training_loss is not re-entrant.
    static float *h_pre = NULL, *h_act = NULL, *o_pre = NULL, *o_act = NULL;
    static size_t h_cap = 0, o_cap = 0;
    size_t h_need = (size_t)max_samples * hid_size * sizeof(float);
    size_t o_need = (size_t)max_samples * out_size * sizeof(float);

    if (h_need > h_cap) {
        free(h_pre); free(h_act);
        h_pre = h_act = NULL;
        h_cap = 0;
        if (posix_memalign((void **)&h_pre, 64, h_need) != 0 ||
            posix_memalign((void **)&h_act, 64, h_need) != 0) {
            free(h_pre); free(h_act);
            h_pre = h_act = NULL;
            return -1.0f;
        }
        h_cap = h_need;
    }
    if (o_need > o_cap) {
        free(o_pre); free(o_act);
        o_pre = o_act = NULL;
        o_cap = 0;
        if (posix_memalign((void **)&o_pre, 64, o_need) != 0 ||
            posix_memalign((void **)&o_act, 64, o_need) != 0) {
            free(o_pre); free(o_act);
            o_pre = o_act = NULL;
            return -1.0f;
        }
        o_cap = o_need;
    }

    float *hidden_weights = nn->hidden.weights;
    float *hidden_biases  = nn->hidden.biases;
    float *output_weights = nn->output.weights;
    float *output_biases  = nn->output.biases;

    double total_se = 0.0;

    for (int start = 0; start < num_samples; start += LOSS_BATCH) {
        int batch_end = (start + LOSS_BATCH < num_samples)
                            ? start + LOSS_BATCH
                            : num_samples;
        int n = batch_end - start;
        int h_batch = n * hid_size;
        int o_batch = n * out_size;

        // Forward pass over the batch
        memset(h_pre, 0, h_batch * sizeof(float));
        mat_mat_mult_transposeB(inputs[start], hidden_weights, h_pre,
                                n, input_size, hid_size);
        add_bias_rowwise(h_pre, hidden_biases, n, hid_size);
        sigmoid_batch_to(h_pre, h_act, h_batch);

        memset(o_pre, 0, o_batch * sizeof(float));
        mat_mat_mult_transposeB(h_act, output_weights, o_pre,
                                n, hid_size, out_size);
        add_bias_rowwise(o_pre, output_biases, n, out_size);
        sigmoid_batch_to(o_pre, o_act, o_batch);

        // Per-sample squared error
        for (int s = 0; s < n; s++) {
            int s_base = s * out_size;
            for (int i = 0; i < out_size; i++) {
                float diff = o_act[s_base + i] - targets[start + s][i];
                total_se += (double)diff * diff;
            }
        }
    }

    #undef LOSS_BATCH

    return (float)(0.5 * total_se / num_samples);
}
