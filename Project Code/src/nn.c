#include "nn.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static NeuralNetwork* nn_allocate(int input_size, int hidden_size,
                                  int output_size) {
    if (input_size <= 0 || hidden_size <= 0 || output_size <= 0)
        return NULL;
    if (input_size > INT_MAX / hidden_size ||
        hidden_size > INT_MAX / output_size)
        return NULL;

    NeuralNetwork* nn = NULL;
    if (posix_memalign((void**)&nn, 64, sizeof(*nn)) != 0)
        return NULL;

    /* Keep unset pointers null so nn_free is safe after partial allocation. */
    memset(nn, 0, sizeof(*nn));

    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;

    hid->input_size  = input_size;
    hid->output_size = hidden_size;
    out->input_size  = hidden_size;
    out->output_size = output_size;

    int w_hid_size = input_size  * hidden_size;
    int w_out_size = hidden_size * output_size;

    #define ALLOC_FLOATS(ptr, count) do { \
        if (posix_memalign((void**)&(ptr), 64, \
                           (size_t)(count) * sizeof(float)) != 0) { \
            nn_free(nn); \
            return NULL; \
        } \
    } while (0)

    ALLOC_FLOATS(hid->weights, w_hid_size);
    ALLOC_FLOATS(hid->weight_grads, w_hid_size);
    ALLOC_FLOATS(hid->biases, hidden_size);
    ALLOC_FLOATS(hid->bias_grads, hidden_size);

    ALLOC_FLOATS(out->weights, w_out_size);
    ALLOC_FLOATS(out->weight_grads, w_out_size);
    ALLOC_FLOATS(out->biases, output_size);
    ALLOC_FLOATS(out->bias_grads, output_size);

    #undef ALLOC_FLOATS

    memset(hid->weight_grads, 0, (size_t)w_hid_size * sizeof(float));
    memset(hid->bias_grads, 0, (size_t)hidden_size * sizeof(float));
    memset(out->weight_grads, 0, (size_t)w_out_size * sizeof(float));
    memset(out->bias_grads, 0, (size_t)output_size * sizeof(float));
    return nn;
}

NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size) {
    NeuralNetwork* nn = nn_allocate(input_size, hidden_size, output_size);
    if (!nn)
        return NULL;

    Layer* hid = &nn->hidden;
    Layer* out = &nn->output;
    int w_hid_size = input_size * hidden_size;
    int w_out_size = hidden_size * output_size;

    /* Xavier-uniform initialization keeps sigmoid pre-activation variance
     * approximately stable as layer widths change. */
    float hid_limit = sqrtf((float)(6.0 / ((double)input_size + hidden_size)));
    float out_limit = sqrtf((float)(6.0 / ((double)hidden_size + output_size)));
    for (int i = 0; i < w_hid_size; i++)
        hid->weights[i] = (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f)
                          * hid_limit;
    for (int i = 0; i < w_out_size; i++)
        out->weights[i] = (2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f)
                          * out_limit;

    memset(hid->biases, 0, (size_t)hidden_size * sizeof(float));
    memset(out->biases, 0, (size_t)output_size * sizeof(float));

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
    }
    free(nn);
}

NeuralNetwork* nn_clone(const NeuralNetwork* src) {
    if (!src) return NULL;

    NeuralNetwork* clone = nn_allocate(src->hidden.input_size,
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
