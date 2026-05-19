#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn.h"

/* Declared in nn.c — not part of the public header but needed by the driver. */
extern void nn_zero_grads(NeuralNetwork* nn);

#define INPUT_SIZE   4
#define HIDDEN_SIZE  8
#define OUTPUT_SIZE  1

#define NUM_SAMPLES  1000
#define BATCH_SIZE   32
#define NUM_EPOCHS   10
#define LEARNING_RATE 0.5f

/*
 * Simple synthetic binary classification rule:
 *   target = (x[0] + x[1] > 1.0) ? 1.0 : 0.0
 * The network must learn a decision boundary in the first two dimensions.
 */
static void generate_dataset(float* inputs, float* targets, int num_samples) {
    for (int i = 0; i < num_samples; i++) {
        float* sample = &inputs[i * INPUT_SIZE];
        for (int j = 0; j < INPUT_SIZE; j++)
            sample[j] = (float)rand() / (float)RAND_MAX;
        /* Use first two features for the decision rule — pattern is learnable. */
        targets[i] = (sample[0] + sample[1] > 1.0f) ? 1.0f : 0.0f;
    }
}

/* Mean Squared Error over the full dataset. */
static float compute_mse(NeuralNetwork* nn, float* inputs, float* targets,
                         int num_samples) {
    float total = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        nn_forward(nn, &inputs[i * INPUT_SIZE]);
        float error = nn->output.activations[0] - targets[i];
        total += error * error;
    }
    return total / (float)num_samples;
}

/* Fisher-Yates shuffle of an index array. */
static void shuffle_indices(int* indices, int num_samples) {
    for (int i = num_samples - 1; i > 0; i--) {
        int swap_idx = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[swap_idx];
        indices[swap_idx] = temp;
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    /* Allocate and populate the synthetic dataset. */
    float* inputs  = (float*)malloc(NUM_SAMPLES * INPUT_SIZE * sizeof(float));
    float* targets = (float*)malloc(NUM_SAMPLES * sizeof(float));
    int*   indices = (int*)malloc(NUM_SAMPLES * sizeof(int));
    if (!inputs || !targets || !indices) {
        fprintf(stderr, "Dataset allocation failed.\n");
        free(inputs); free(targets); free(indices);
        return 1;
    }
    generate_dataset(inputs, targets, NUM_SAMPLES);
    for (int i = 0; i < NUM_SAMPLES; i++)
        indices[i] = i;

    /* Create the network. */
    NeuralNetwork* nn = nn_create(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    if (!nn) {
        fprintf(stderr, "Network creation failed.\n");
        free(inputs); free(targets); free(indices);
        return 1;
    }

    float lr_per_sample = LEARNING_RATE / (float)BATCH_SIZE;
    float epoch_1_loss   = 0.0f;
    float epoch_10_loss  = 0.0f;

    printf("=== Sequential Feedforward NN Training ===\n");
    printf("Architecture: %d -> %d -> %d\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    printf("Samples: %d | Batch: %d | Epochs: %d | LR: %.4f\n\n",
           NUM_SAMPLES, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE);

    for (int epoch = 1; epoch <= NUM_EPOCHS; epoch++) {
        /* Shuffle sample order each epoch for stochasticity. */
        shuffle_indices(indices, NUM_SAMPLES);

        /*
         * Mini-batch gradient descent: iterate over batches, accumulate
         * gradients per batch, then update weights once per batch.
         */
        for (int start = 0; start < NUM_SAMPLES; start += BATCH_SIZE) {
            int batch_end = (start + BATCH_SIZE < NUM_SAMPLES)
                            ? start + BATCH_SIZE : NUM_SAMPLES;

            nn_zero_grads(nn);

            /* Accumulate gradients over the mini-batch. */
            for (int batch_idx = start; batch_idx < batch_end; batch_idx++) {
                int sample_idx = indices[batch_idx];
                nn_forward(nn, &inputs[sample_idx * INPUT_SIZE]);
                nn_backward(nn, &inputs[sample_idx * INPUT_SIZE],
                            &targets[sample_idx]);
            }

            nn_update_weights(nn, lr_per_sample);
        }

        /* Evaluate loss on the full dataset after each epoch. */
        float loss = compute_mse(nn, inputs, targets, NUM_SAMPLES);
        printf("Epoch %2d | MSE Loss: %.6f\n", epoch, loss);

        if (epoch == 1)  epoch_1_loss  = loss;
        if (epoch == 10) epoch_10_loss = loss;
    }

    /*
     * VERIFICATION TEST: Confirm the network learned by checking that
     * epoch-10 loss is strictly lower than epoch-1 loss.
     */
    printf("\n=== Verification ===\n");
    printf("Epoch  1 loss: %.6f\n", epoch_1_loss);
    printf("Epoch 10 loss: %.6f\n", epoch_10_loss);

    if (epoch_10_loss < epoch_1_loss) {
        printf("[VERIFICATION SUCCESS]: Network is learning and loss is decreasing.\n");
    } else {
        printf("[VERIFICATION FAILURE]: Loss did not decrease — network is not learning.\n");
        nn_free(nn);
        free(inputs); free(targets); free(indices);
        return 1;
    }

    nn_free(nn);
    free(inputs);
    free(targets);
    free(indices);
    return 0;
}
