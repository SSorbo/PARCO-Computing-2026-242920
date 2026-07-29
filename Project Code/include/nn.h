#ifndef NN_H
#define NN_H

typedef struct {
    int input_size;
    int output_size;
    /* Output-major row storage: weights[i * input_size + j] connects
     * input j to output i. */
    float *weights;
    float *weight_grads;
    float *biases;
    float *bias_grads;
} Layer;

typedef struct {
    Layer hidden;
    Layer output;
} NeuralNetwork;

NeuralNetwork* nn_create(int input_size, int hidden_size, int output_size);
void nn_free(NeuralNetwork* nn);

/* Clone parameters into an independent model for repeatable benchmark runs. */
NeuralNetwork* nn_clone(const NeuralNetwork* src);

#endif
