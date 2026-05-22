#include <stdlib.h>
#include <string.h>
#include "nn.h"
#include "math_utils.h"

void nn_forward(NeuralNetwork* nn, float* input) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;

    int in_size  = hid->input_size;
    int hid_size = hid->output_size;
    int out_size = out->output_size;

    mat_vec_mult(hid->weights, input, hid->biases, hid->pre_acts,
                 hid_size, in_size);
    for (int j = 0; j < hid_size; j++)
        hid->activations[j] = sigmoid(hid->pre_acts[j]);

    mat_vec_mult(out->weights, hid->activations, out->biases, out->pre_acts,
                 out_size, hid_size);
    for (int j = 0; j < out_size; j++)
        out->activations[j] = sigmoid(out->pre_acts[j]);
}

void nn_backward(NeuralNetwork* nn, float* input, float* target) {
    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;

    int hid_size = hid->output_size;
    int out_size = out->output_size;
    int in_size  = hid->input_size;

    for (int j = 0; j < out_size; j++) {
        float error = out->activations[j] - target[j];
        out->deltas[j] = error * sigmoid_deriv(out->pre_acts[j]);
    }

    for (int i = 0; i < hid_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < out_size; j++) {
            sum += out->weights[j * hid_size + i] * out->deltas[j];
        }
        hid->deltas[i] = sum * sigmoid_deriv(hid->pre_acts[i]);
    }

    for (int j = 0; j < out_size; j++) {
        float delta = out->deltas[j];
        int base = j * hid_size;
        for (int i = 0; i < hid_size; i++) {
            out->weight_grads[base + i] += hid->activations[i] * delta;
        }
    }
    for (int j = 0; j < out_size; j++)
        out->bias_grads[j] += out->deltas[j];

    for (int j = 0; j < hid_size; j++) {
        float delta = hid->deltas[j];
        int base = j * in_size;
        for (int i = 0; i < in_size; i++) {
            hid->weight_grads[base + i] += input[i] * delta;
        }
    }
    for (int j = 0; j < hid_size; j++)
        hid->bias_grads[j] += hid->deltas[j];
}

void train_sequential(NeuralNetwork* nn,
                      float** inputs,
                      float** targets,
                      int num_samples,
                      int epochs,
                      int batch_size) {
    if (!nn || !inputs || !targets || num_samples <= 0 || epochs <= 0 || batch_size <= 0)
        return;

    int* indices = (int*)malloc(num_samples * sizeof(int));
    if (!indices)
        return;

    for (int i = 0; i < num_samples; i++)
        indices[i] = i;

    const float learning_rate = 0.5f;
    const float lr_per_sample = learning_rate / (float)batch_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int start = 0; start < num_samples; start += batch_size) {
            int batch_end = (start + batch_size < num_samples) ? start + batch_size : num_samples;

            nn_zero_grads(nn);
            for (int sample = start; sample < batch_end; sample++) {
                int sample_idx = indices[sample];
                nn_forward(nn, inputs[sample_idx]);
                nn_backward(nn, inputs[sample_idx], targets[sample_idx]);
            }

            nn_update_weights(nn, lr_per_sample);
        }
    }

    free(indices);
}
