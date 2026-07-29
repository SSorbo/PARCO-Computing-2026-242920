#include "nn.h"
#include "test_common.h"
#include "test_suites.h"
#include <stdint.h>

/*
 * Memory-safety tests covering allocation edge cases, alignment
 * guarantees, and clone independence.
 */

/* ---- 1. Allocation edge cases ---- */

static int test_null_on_zero_dims(void) {
    int fail = 0;

    NeuralNetwork* n0 = nn_create(0, 5, 5);
    if (n0 != NULL) { printf("  nn_create(0,5,5) should return NULL\n"); fail++; nn_free(n0); }

    NeuralNetwork* n1 = nn_create(5, 0, 5);
    if (n1 != NULL) { printf("  nn_create(5,0,5) should return NULL\n"); fail++; nn_free(n1); }

    NeuralNetwork* n2 = nn_create(5, 5, 0);
    if (n2 != NULL) { printf("  nn_create(5,5,0) should return NULL\n"); fail++; nn_free(n2); }

    return fail;
}

static int test_nn_free_null(void) {
    nn_free(NULL);  /* must not crash */
    return 0;
}

static int test_nn_clone_null(void) {
    NeuralNetwork* c = nn_clone(NULL);
    if (c != NULL) { printf("  nn_clone(NULL) should return NULL\n"); return 1; }
    return 0;
}

/* ---- 2. Alignment verification ---- */

static int check_alignment(const void* ptr, const char* label) {
    if (ptr == NULL) {
        printf("  %s is NULL\n", label);
        return 1;
    }
    if (((uintptr_t)ptr) % 64 != 0) {
        printf("  %s not 64-byte aligned: %p\n", label, (void*)ptr);
        return 1;
    }
    return 0;
}

static int test_nn_alignment(void) {
    NeuralNetwork* nn = nn_create(4, 3, 2);
    if (!nn) { printf("  nn_create failed\n"); return 1; }

    int fail = 0;
    Layer* layers[2] = { &nn->hidden, &nn->output };
    const char* names[] = {"hidden", "output"};
    const char* fields[] = {"weights", "weight_grads", "biases", "bias_grads"};

    for (int li = 0; li < 2; li++) {
        float** ptrs = (float*[]) {
            layers[li]->weights, layers[li]->weight_grads, layers[li]->biases,
            layers[li]->bias_grads
        };
        for (int fi = 0; fi < 4; fi++) {
            char label[64];
            snprintf(label, sizeof(label), "%s.%s", names[li], fields[fi]);
            fail += check_alignment(ptrs[fi], label);
        }
    }

    nn_free(nn);
    return fail;
}

static int test_nn_struct_alignment(void) {
    /* nn_allocate requests 64-byte alignment for the model structure. */
    NeuralNetwork* nn = nn_create(2, 2, 2);
    if (!nn) return 1;

    int fail = check_alignment(nn, "NeuralNetwork struct");
    nn_free(nn);
    return fail;
}

/* ---- 3. Clone independence ---- */

static int test_clone_independence(void) {
    NeuralNetwork* orig = nn_create(3, 4, 2);
    if (!orig) return 1;

    NeuralNetwork* clone = nn_clone(orig);
    if (!clone) { nn_free(orig); return 1; }

    int fail = 0;

    /* 3a. Verify initial equality */
    int w_hid = 3 * 4;
    int w_out = 4 * 2;
    if (!arrays_bitwise_equal(orig->hidden.weights, clone->hidden.weights, w_hid)) {
        printf("  clone weights should be identical to original\n"); fail++;
    }
    if (!arrays_bitwise_equal(orig->output.weights, clone->output.weights, w_out)) {
        printf("  clone output weights should be identical to original\n"); fail++;
    }

    /* 3b. Modify clone — original must be unchanged */
    clone->hidden.weights[0] = 999.0f;
    if (orig->hidden.weights[0] == 999.0f) {
        printf("  modifying clone affected original (shared pointer!)\n"); fail++;
    }

    /* 3c. Modify clone bias — original must be unchanged */
    clone->output.biases[0] = -555.0f;
    if (orig->output.biases[0] == -555.0f) {
        printf("  modifying clone bias affected original (shared pointer!)\n"); fail++;
    }

    /* 3d. Pointers must differ (independent allocations) */
    if (orig->hidden.weights == clone->hidden.weights) {
        printf("  hidden weights pointer shared between clone and original\n"); fail++;
    }
    if (orig->output.biases == clone->output.biases) {
        printf("  output biases pointer shared between clone and original\n"); fail++;
    }

    nn_free(orig);
    nn_free(clone);
    return fail;
}

static int test_zero_initial_gradients(void) {
    NeuralNetwork* nn = nn_create(4, 3, 2);
    if (!nn) return 1;

    int fail = 0;
    int w_hid = 4 * 3;
    int w_out = 3 * 2;

    for (int i = 0; i < w_hid; i++)
        if (nn->hidden.weight_grads[i] != 0.0f) { fail++; break; }
    for (int i = 0; i < 3; i++)
        if (nn->hidden.bias_grads[i] != 0.0f) { fail++; break; }
    for (int i = 0; i < w_out; i++)
        if (nn->output.weight_grads[i] != 0.0f) { fail++; break; }
    for (int i = 0; i < 2; i++)
        if (nn->output.bias_grads[i] != 0.0f) { fail++; break; }

    if (fail) printf("  gradients not zero-initialized\n");
    nn_free(nn);
    return fail;
}

/* ---- public entry point ---- */

int run_memory_safety_tests(void) {
    int total = 0;

    printf("\n=== Memory Safety Tests ===\n");

    printf("\n--- Allocation edge cases ---\n");
    int f1 = test_null_on_zero_dims();
    int f2 = test_nn_free_null();
    int f3 = test_nn_clone_null();
    printf("  zero-dim returns NULL:  %s\n", f1 == 0 ? "PASS" : "FAIL");
    printf("  nn_free(NULL) no-crash: %s\n", f2 == 0 ? "PASS" : "FAIL");
    printf("  nn_clone(NULL) -> NULL: %s\n", f3 == 0 ? "PASS" : "FAIL");

    printf("\n--- Alignment verification ---\n");
    int f4 = test_nn_alignment();
    int f5 = test_nn_struct_alignment();
    printf("  all fields 64-byte aligned: %s\n", f4 == 0 ? "PASS" : "FAIL");
    printf("  struct 64-byte aligned:     %s\n", f5 == 0 ? "PASS" : "FAIL");

    printf("\n--- Clone independence ---\n");
    int f6 = test_clone_independence();
    int f7 = test_zero_initial_gradients();
    printf("  clone independence:     %s\n", f6 == 0 ? "PASS" : "FAIL");
    printf("  zero initial gradients: %s\n", f7 == 0 ? "PASS" : "FAIL");

    int failed = f1 + f2 + f3 + f4 + f5 + f6 + f7;
    total = 7;
    printf("\nMemory safety: %d/%d passed\n", total - failed, total);
    return failed;
}
