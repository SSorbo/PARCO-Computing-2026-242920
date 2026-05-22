#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "nn.h"

/* Forward declarations for the extracted training modules. */
void train_sequential(NeuralNetwork* nn,
                      float** inputs,
                      float** targets,
                      int num_samples,
                      int epochs,
                      int batch_size);

void train_openmp(NeuralNetwork* nn,
                  float** inputs,
                  float** targets,
                  int num_samples,
                  int epochs,
                  int batch_size);

static void generate_dataset(float** inputs,
                             float** targets,
                             int num_samples,
                             int input_size,
                             int output_size) {
    for (int sample = 0; sample < num_samples; sample++) {
        for (int feature = 0; feature < input_size; feature++) {
            inputs[sample][feature] = (float)rand() / (float)RAND_MAX;
        }

        int label = 0;
        float best = inputs[sample][0];
        for (int k = 1; k < output_size; k++) {
            if (inputs[sample][k] > best) {
                best = inputs[sample][k];
                label = k;
            }
        }

        for (int k = 0; k < output_size; k++) {
            targets[sample][k] = (k == label) ? 1.0f : 0.0f;
        }
    }
}

int main(void) {
    const int num_samples = 2048;
    const int input_size = 784;
    const int hidden_size = 256;
    const int output_size = 10;
    const int epochs = 10;
    const int batch_size = 64;

    srand(22222);

    float* input_data = NULL;
    float* target_data = NULL;
    if (posix_memalign((void**)&input_data, 64,
                       (size_t)num_samples * input_size * sizeof(float)) != 0 ||
        posix_memalign((void**)&target_data, 64,
                       (size_t)num_samples * output_size * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: dataset allocation failed.\n");
        free(input_data);
        free(target_data);
        return 1;
    }

    float** inputs = (float**)malloc(num_samples * sizeof(float*));
    float** targets = (float**)malloc(num_samples * sizeof(float*));
    if (!inputs || !targets) {
        fprintf(stderr, "ERROR: dataset pointer allocation failed.\n");
        free(input_data);
        free(target_data);
        free(inputs);
        free(targets);
        return 1;
    }

    for (int i = 0; i < num_samples; i++) {
        inputs[i] = &input_data[(size_t)i * input_size];
        targets[i] = &target_data[(size_t)i * output_size];
    }

    generate_dataset(inputs, targets, num_samples, input_size, output_size);

    NeuralNetwork* master_nn = nn_create(input_size, hidden_size, output_size);
    if (!master_nn) {
        fprintf(stderr, "ERROR: failed to create master neural network.\n");
        free(inputs);
        free(targets);
        free(input_data);
        free(target_data);
        return 1;
    }

    NeuralNetwork* seq_nn = nn_clone(master_nn);
    NeuralNetwork* omp_nn = nn_clone(master_nn);
    if (!seq_nn || !omp_nn) {
        fprintf(stderr, "ERROR: failed to clone neural network state.\n");
        nn_free(master_nn);
        nn_free(seq_nn);
        nn_free(omp_nn);
        free(inputs);
        free(targets);
        free(input_data);
        free(target_data);
        return 1;
    }

    printf("=== Benchmark Harness ===\n");
    printf("Samples: %d | Input: %d | Hidden: %d | Output: %d | Batch: %d | Epochs: %d\n",
           num_samples, input_size, hidden_size, output_size, batch_size, epochs);

    double start = omp_get_wtime();
    train_sequential(seq_nn, inputs, targets, num_samples, epochs, batch_size);
    double time_seq = omp_get_wtime() - start;
    printf("Sequential training time: %.6f seconds\n", time_seq);

    start = omp_get_wtime();
    train_openmp(omp_nn, inputs, targets, num_samples, epochs, batch_size);
    double time_omp = omp_get_wtime() - start;
    printf("OpenMP training time: %.6f seconds\n", time_omp);

    if (time_omp > 0.0) {
        printf("Speedup: %.2fx\n", time_seq / time_omp);
    }

    verify_identical_weights(seq_nn, omp_nn);

    nn_free(master_nn);
    nn_free(seq_nn);
    nn_free(omp_nn);
    free(inputs);
    free(targets);
    free(input_data);
    free(target_data);

    return 0;
}
