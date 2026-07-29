#include "nn.h"
#include "train.h"
#include "test_common.h"
#include "test_suites.h"
#include <omp.h>

/*
 * Training reproducibility tests.
 *
 * These verify that:
 * 1. Sequential training is fully deterministic.
 * 2. OpenMP training with fixed thread count is deterministic
 *    (within relaxed tolerance for non-associative parallel reductions).
 * 3. Sequential and OpenMP training converge to the same result.
 */

/* Minimal dataset for fast training */
#define REPRO_NUM_SAMPLES 32
#define REPRO_INPUT_SIZE   4
#define REPRO_HIDDEN_SIZE  3
#define REPRO_OUTPUT_SIZE  2
#define REPRO_EPOCHS       5
#define REPRO_BATCH_SIZE   8

/* ---- dataset generation (matches the benchmark drivers) ---- */

static void generate_small_dataset(float** inputs, float** targets) {
    for (int sample = 0; sample < REPRO_NUM_SAMPLES; sample++) {
        for (int feature = 0; feature < REPRO_INPUT_SIZE; feature++) {
            inputs[sample][feature] = (float)rand() / (float)RAND_MAX;
        }
        int label = 0;
        float best = inputs[sample][0];
        for (int k = 1; k < REPRO_OUTPUT_SIZE; k++) {
            if (inputs[sample][k] > best) {
                best = inputs[sample][k];
                label = k;
            }
        }
        for (int k = 0; k < REPRO_OUTPUT_SIZE; k++) {
            targets[sample][k] = (k == label) ? 1.0f : 0.0f;
        }
    }
}

/* ---- compare two networks' weights and biases ---- */

static int networks_equal(NeuralNetwork* a, NeuralNetwork* b, float tol) {
    int w_hid = a->hidden.input_size * a->hidden.output_size;
    int w_out = a->output.input_size * a->output.output_size;
    int fail = 0;

    if (compare_arrays(a->hidden.weights, b->hidden.weights, w_hid, tol) > 0) {
        printf("  hidden weights differ\n"); fail = 1;
    }
    if (compare_arrays(a->hidden.biases, b->hidden.biases,
                       a->hidden.output_size, tol) > 0) {
        printf("  hidden biases differ\n"); fail = 1;
    }
    if (compare_arrays(a->output.weights, b->output.weights, w_out, tol) > 0) {
        printf("  output weights differ\n"); fail = 1;
    }
    if (compare_arrays(a->output.biases, b->output.biases,
                       a->output.output_size, tol) > 0) {
        printf("  output biases differ\n"); fail = 1;
    }
    return fail;
}

/* ---- 1. Sequential determinism ---- */

static int test_sequential_determinism(float** inputs, float** targets) {
    /* Two networks from the same master, trained identically */
    srand(12345);
    NeuralNetwork* master = nn_create(REPRO_INPUT_SIZE, REPRO_HIDDEN_SIZE, REPRO_OUTPUT_SIZE);
    if (!master) return 1;

    NeuralNetwork* nn1 = nn_clone(master);
    NeuralNetwork* nn2 = nn_clone(master);
    nn_free(master);
    if (!nn1 || !nn2) { nn_free(nn1); nn_free(nn2); return 1; }

    if (train_sequential(nn1, inputs, targets, REPRO_NUM_SAMPLES,
                            REPRO_EPOCHS, REPRO_BATCH_SIZE, 0) < 0.0 ||
        train_sequential(nn2, inputs, targets, REPRO_NUM_SAMPLES,
                            REPRO_EPOCHS, REPRO_BATCH_SIZE, 0) < 0.0) {
        nn_free(nn1);
        nn_free(nn2);
        return 1;
    }

    int fail = networks_equal(nn1, nn2, TEST_TOLERANCE);
    if (!fail) {
        /* Also check bitwise identity — sequential should be exactly deterministic */
        int w_hid = REPRO_INPUT_SIZE * REPRO_HIDDEN_SIZE;
        int w_out = REPRO_HIDDEN_SIZE * REPRO_OUTPUT_SIZE;
        int bitwise = 1;
        bitwise &= arrays_bitwise_equal(nn1->hidden.weights, nn2->hidden.weights, w_hid);
        bitwise &= arrays_bitwise_equal(nn1->output.weights, nn2->output.weights, w_out);
        bitwise &= arrays_bitwise_equal(nn1->hidden.biases, nn2->hidden.biases, REPRO_HIDDEN_SIZE);
        bitwise &= arrays_bitwise_equal(nn1->output.biases, nn2->output.biases, REPRO_OUTPUT_SIZE);
        if (!bitwise) {
            printf("  sequential training not bitwise-identical (non-deterministic!)\n");
            fail = 1;
        }
    }

    nn_free(nn1);
    nn_free(nn2);
    return fail;
}

/* ---- 2. OpenMP determinism (fixed thread count) ---- */

static int test_openmp_determinism(float** inputs, float** targets, int num_threads) {
    omp_set_num_threads(num_threads);

    srand(12345);
    NeuralNetwork* master = nn_create(REPRO_INPUT_SIZE, REPRO_HIDDEN_SIZE, REPRO_OUTPUT_SIZE);
    if (!master) return 1;

    NeuralNetwork* nn1 = nn_clone(master);
    NeuralNetwork* nn2 = nn_clone(master);
    nn_free(master);
    if (!nn1 || !nn2) { nn_free(nn1); nn_free(nn2); return 1; }

    if (train_openmp(nn1, inputs, targets, REPRO_NUM_SAMPLES,
                        REPRO_EPOCHS, REPRO_BATCH_SIZE, 0) < 0.0 ||
        train_openmp(nn2, inputs, targets, REPRO_NUM_SAMPLES,
                        REPRO_EPOCHS, REPRO_BATCH_SIZE, 0) < 0.0) {
        nn_free(nn1);
        nn_free(nn2);
        return 1;
    }

    /*
     * Parallel bias reductions use a deterministic order for a fixed team,
     * but floating-point grouping differs from the sequential reduction.
     * Keep a relaxed tolerance for biases.
     */
    int fail = 0;
    int w_hid = REPRO_INPUT_SIZE * REPRO_HIDDEN_SIZE;
    int w_out = REPRO_HIDDEN_SIZE * REPRO_OUTPUT_SIZE;

    if (compare_arrays(nn1->hidden.weights, nn2->hidden.weights, w_hid, TEST_TOLERANCE) > 0)
        fail = 1;
    if (compare_arrays(nn1->hidden.biases, nn2->hidden.biases,
                       REPRO_HIDDEN_SIZE, TEST_TOLERANCE_LOOSE) > 0)
        fail = 1;
    if (compare_arrays(nn1->output.weights, nn2->output.weights, w_out, TEST_TOLERANCE) > 0)
        fail = 1;
    if (compare_arrays(nn1->output.biases, nn2->output.biases,
                       REPRO_OUTPUT_SIZE, TEST_TOLERANCE_LOOSE) > 0)
        fail = 1;

    nn_free(nn1);
    nn_free(nn2);
    return fail;
}

/* ---- 3. Seq vs. OpenMP convergence ---- */

static int test_seq_vs_openmp(float** inputs, float** targets) {
    srand(12345);
    NeuralNetwork* master = nn_create(REPRO_INPUT_SIZE, REPRO_HIDDEN_SIZE, REPRO_OUTPUT_SIZE);
    if (!master) return 1;

    NeuralNetwork* seq_nn = nn_clone(master);
    NeuralNetwork* omp_nn = nn_clone(master);
    nn_free(master);
    if (!seq_nn || !omp_nn) { nn_free(seq_nn); nn_free(omp_nn); return 1; }

    if (train_sequential(seq_nn, inputs, targets, REPRO_NUM_SAMPLES,
                            REPRO_EPOCHS, REPRO_BATCH_SIZE, 0) < 0.0 ||
        train_openmp(omp_nn, inputs, targets, REPRO_NUM_SAMPLES,
                        REPRO_EPOCHS, REPRO_BATCH_SIZE, 0) < 0.0) {
        nn_free(seq_nn);
        nn_free(omp_nn);
        return 1;
    }

    int fail = networks_equal(seq_nn, omp_nn, TEST_TOLERANCE);

    nn_free(seq_nn);
    nn_free(omp_nn);
    return fail;
}

/* ---- public entry point ---- */

int run_reproducibility_tests(void) {
    int fail = 0;

    printf("\n=== Training Reproducibility Tests ===\n");
    printf("Network: %d-%d-%d, samples=%d, epochs=%d, batch=%d\n",
           REPRO_INPUT_SIZE, REPRO_HIDDEN_SIZE, REPRO_OUTPUT_SIZE,
           REPRO_NUM_SAMPLES, REPRO_EPOCHS, REPRO_BATCH_SIZE);

    /* Build dataset once */
    float* input_data  = aligned_alloc_float(REPRO_NUM_SAMPLES * REPRO_INPUT_SIZE);
    float* target_data = aligned_alloc_float(REPRO_NUM_SAMPLES * REPRO_OUTPUT_SIZE);
    float** inputs  = malloc(REPRO_NUM_SAMPLES * sizeof(float*));
    float** targets = malloc(REPRO_NUM_SAMPLES * sizeof(float*));
    if (!inputs || !targets) {
        fprintf(stderr, "FATAL: reproducibility dataset allocation failed\n");
        free(input_data); free(target_data); free(inputs); free(targets);
        return 1;
    }
    for (int i = 0; i < REPRO_NUM_SAMPLES; i++) {
        inputs[i]  = &input_data[(size_t)i * REPRO_INPUT_SIZE];
        targets[i] = &target_data[(size_t)i * REPRO_OUTPUT_SIZE];
    }

    srand(42);
    generate_small_dataset(inputs, targets);

    /* Test 1: Sequential determinism */
    printf("\n--- Sequential training determinism ---\n");
    int f1 = test_sequential_determinism(inputs, targets);
    printf("  %s (must be bitwise-identical)\n", f1 == 0 ? "PASS" : "FAIL");
    fail += f1;

    /* Test 2: OpenMP determinism with fixed threads */
    printf("\n--- OpenMP training determinism (2 threads) ---\n");
    int f2 = test_openmp_determinism(inputs, targets, 2);
    printf("  %s (within float tolerance)\n", f2 == 0 ? "PASS" : "FAIL");
    fail += f2;

    printf("\n--- OpenMP training determinism (4 threads) ---\n");
    int f3 = test_openmp_determinism(inputs, targets, 4);
    printf("  %s (within float tolerance)\n", f3 == 0 ? "PASS" : "FAIL");
    fail += f3;

    /* Test 3: Seq vs. OpenMP convergence */
    printf("\n--- Sequential vs. OpenMP convergence ---\n");
    int f4 = test_seq_vs_openmp(inputs, targets);
    printf("  %s (within 1e-4 tolerance)\n", f4 == 0 ? "PASS" : "FAIL");
    fail += f4;

    /* Cleanup */
    free(input_data);
    free(target_data);
    free(inputs);
    free(targets);

    printf("\nReproducibility: %d failures\n", fail);
    return fail;
}
