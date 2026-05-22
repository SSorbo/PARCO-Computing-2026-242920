#include <stdlib.h>
#include <string.h>
#include "nn.h"

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
