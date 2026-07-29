#include "nn.h"
#include "math_utils.h"
#include "test_common.h"
#include "test_suites.h"
#include <string.h>

/*
 * Finite-difference gradient verification for backprop.
 *
 * For each weight and bias in a tiny 2-layer network, we compare the
 * analytical gradient from backprop against a central-difference
 * numerical approximation:
 *
 *   numerical = (loss(w+eps) - loss(w-eps)) / (2*eps)
 *
 * Tolerances are calibrated for float32.  Central differences with
 * eps=1e-3 give O(1e-6) truncation error.  The dominant noise source
 * is the subtraction (loss_plus - loss_minus), where float32 roundoff
 * in the loss (~1e-7) is amplified by 1/(2*eps) ≈ 500, producing
 * noise of ~5e-5 in the gradient.  We use conservative thresholds
 * well above this noise floor.
 *
 * Real backprop bugs (wrong sign, missing derivative, dimension
 * mismatch) produce errors orders of magnitude larger — typically
 * 1e-1 or more — so these thresholds catch the targeted defect classes.
 */

#define EPSILON     1e-3f
#define GRAD_ABS_TOL 5e-4f
#define GRAD_REL_TOL 1e-2f

/* ---- half-squared-error objective ---- */
static float compute_loss(const float* output, const float* target, int n) {
    float loss = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = output[i] - target[i];
        loss += 0.5f * diff * diff;
    }
    return loss;
}

/* ---- small test-local reference helpers ---- */
static void add_bias_rowwise(float* matrix, const float* bias,
                              int batch_count, int cols) {
    for (int sample = 0; sample < batch_count; sample++) {
        for (int col = 0; col < cols; col++) {
            matrix[(size_t)sample * cols + col] += bias[col];
        }
    }
}

static void sigmoid_batch(const float* src, float* dst, int count) {
    for (int i = 0; i < count; i++)
        dst[i] = sigmoid(src[i]);
}

static void accumulate_bias_grads(const float* deltas, float* bias_grads,
                                   int batch_count, int cols) {
    for (int col = 0; col < cols; col++) {
        float sum = 0.0f;
        for (int sample = 0; sample < batch_count; sample++)
            sum += deltas[(size_t)sample * cols + col];
        bias_grads[col] = sum;  /* overwrite, not accumulate, for test clarity */
    }
}

/*
 * Run one forward pass and return loss.
 * Writes activations into pre-allocated buffers for use by backprop.
 */
static float forward_pass(float* batch_inputs,
                           const float* batch_targets,
                           NeuralNetwork* nn,
                           int batch_count,
                           float* h_pre, float* h_act,
                           float* o_pre, float* o_act) {
    int in_size  = nn->hidden.input_size;
    int hid_size = nn->hidden.output_size;
    int out_size = nn->output.output_size;

    int hid_batch = batch_count * hid_size;
    int out_batch = batch_count * out_size;

    /* hidden layer */
    memset(h_pre, 0, (size_t)hid_batch * sizeof(float));
    mat_mat_mult_transposeB(batch_inputs, nn->hidden.weights, h_pre,
                             batch_count, in_size, hid_size);
    add_bias_rowwise(h_pre, nn->hidden.biases, batch_count, hid_size);
    sigmoid_batch(h_pre, h_act, hid_batch);

    /* output layer */
    memset(o_pre, 0, (size_t)out_batch * sizeof(float));
    mat_mat_mult_transposeB(h_act, nn->output.weights, o_pre,
                             batch_count, hid_size, out_size);
    add_bias_rowwise(o_pre, nn->output.biases, batch_count, out_size);
    sigmoid_batch(o_pre, o_act, out_batch);

    return compute_loss(o_act, batch_targets, out_batch);
}

/*
 * Run one backward pass from the pre-computed activations.
 * Writes analytical gradients into weight_grads / bias_grads.
 * These OVERWRITE (not accumulate) the grad arrays for test clarity.
 */
static void backward_pass(float* batch_inputs,
                           const float* batch_targets,
                           NeuralNetwork* nn,
                           int batch_count,
                           float* h_pre, float* h_act,
                           float* o_pre, float* o_act) {
    int in_size  = nn->hidden.input_size;
    int hid_size = nn->hidden.output_size;
    int out_size = nn->output.output_size;

    int hid_batch = batch_count * hid_size;
    int out_batch = batch_count * out_size;

    float* o_deltas = aligned_alloc_float(out_batch);
    float* h_deltas = aligned_alloc_float(hid_batch);

    /* output deltas: (o_act - target) * sigmoid'(o_pre) */
    for (int i = 0; i < out_batch; i++) {
        float error = o_act[i] - batch_targets[i];
        o_deltas[i] = error * sigmoid_deriv(o_pre[i]);
    }

    /* hidden deltas: (o_deltas * output_weights) * sigmoid'(h_pre) */
    memset(h_deltas, 0, (size_t)hid_batch * sizeof(float));
    mat_mat_mult(o_deltas, nn->output.weights, h_deltas,
                 batch_count, out_size, hid_size);
    for (int i = 0; i < hid_batch; i++)
        h_deltas[i] *= sigmoid_deriv(h_pre[i]);

    /* output weight gradients: o_deltas^T * h_act */
    int w_out_size = out_size * hid_size;
    memset(nn->output.weight_grads, 0, (size_t)w_out_size * sizeof(float));
    mat_mat_mult_transposeA(o_deltas, h_act, nn->output.weight_grads,
                             batch_count, out_size, hid_size);
    accumulate_bias_grads(o_deltas, nn->output.bias_grads, batch_count, out_size);

    /* hidden weight gradients: h_deltas^T * inputs */
    int w_hid_size = hid_size * in_size;
    memset(nn->hidden.weight_grads, 0, (size_t)w_hid_size * sizeof(float));
    mat_mat_mult_transposeA(h_deltas, batch_inputs, nn->hidden.weight_grads,
                             batch_count, hid_size, in_size);
    accumulate_bias_grads(h_deltas, nn->hidden.bias_grads, batch_count, hid_size);

    free(o_deltas);
    free(h_deltas);
}

/*
 * Check a single weight: perturb by +/- epsilon, compute numerical
 * gradient, compare to analytical gradient in grad_analytical.
 */
static int check_weight_gradient(NeuralNetwork* nn,
                                  float* batch_inputs,
                                  const float* batch_targets,
                                  int batch_count,
                                  float* weight_ptr,
                                  float grad_analytical,
                                  const char* label) {
    float original = *weight_ptr;

    /* loss at w + epsilon */
    *weight_ptr = original + EPSILON;
    float* h_pre_p = aligned_alloc_float(batch_count * nn->hidden.output_size);
    float* h_act_p = aligned_alloc_float(batch_count * nn->hidden.output_size);
    float* o_pre_p = aligned_alloc_float(batch_count * nn->output.output_size);
    float* o_act_p = aligned_alloc_float(batch_count * nn->output.output_size);
    float loss_plus = forward_pass(batch_inputs, batch_targets, nn, batch_count,
                                    h_pre_p, h_act_p, o_pre_p, o_act_p);

    /* loss at w - epsilon */
    *weight_ptr = original - EPSILON;
    float* h_pre_m = aligned_alloc_float(batch_count * nn->hidden.output_size);
    float* h_act_m = aligned_alloc_float(batch_count * nn->hidden.output_size);
    float* o_pre_m = aligned_alloc_float(batch_count * nn->output.output_size);
    float* o_act_m = aligned_alloc_float(batch_count * nn->output.output_size);
    float loss_minus = forward_pass(batch_inputs, batch_targets, nn, batch_count,
                                     h_pre_m, h_act_m, o_pre_m, o_act_m);

    /* restore original */
    *weight_ptr = original;

    float grad_numerical = (loss_plus - loss_minus) / (2.0f * EPSILON);

    free(h_pre_p); free(h_act_p); free(o_pre_p); free(o_act_p);
    free(h_pre_m); free(h_act_m); free(o_pre_m); free(o_act_m);

    float abs_diff = fabsf(grad_analytical - grad_numerical);

    /* Absolute tolerance: catches the case where gradients are small
     * and relative error is dominated by float32 finite-difference noise. */
    if (abs_diff < GRAD_ABS_TOL)
        return 0;

    /* Relative tolerance: for larger gradients. */
    float denom = fmaxf(fmaxf(fabsf(grad_analytical), fabsf(grad_numerical)), 1e-8f);
    float rel_err = abs_diff / denom;

    if (rel_err > GRAD_REL_TOL) {
        printf("  [%s] analytical=%.8f numerical=%.8f abs_diff=%.2e rel_err=%.2e\n",
               label, grad_analytical, grad_numerical, (double)abs_diff, (double)rel_err);
        return 1;
    }
    return 0;
}

/*
 * Run the full gradient check for a network with given batch_size.
 */
static int check_all_gradients(int input_size, int hidden_size, int output_size,
                                int batch_size) {
    /* Create network and fixed input/target data */
    NeuralNetwork* nn = nn_create(input_size, hidden_size, output_size);
    if (!nn) { printf("  nn_create failed\n"); return 1; }

    int in_batch  = batch_size * input_size;
    int out_batch = batch_size * output_size;
    float* batch_inputs  = aligned_alloc_float(in_batch);
    float* batch_targets = aligned_alloc_float(out_batch);

    fill_random_positive(batch_inputs, in_batch);
    fill_random_positive(batch_targets, out_batch);

    int hid_size = hidden_size;
    int hid_batch = batch_size * hid_size;

    /* Allocate forward pass buffers */
    float* h_pre = aligned_alloc_float(hid_batch);
    float* h_act = aligned_alloc_float(hid_batch);
    float* o_pre = aligned_alloc_float(out_batch);
    float* o_act = aligned_alloc_float(out_batch);

    /* Run forward pass */
    forward_pass(batch_inputs, batch_targets, nn, batch_size,
                 h_pre, h_act, o_pre, o_act);

    /* Run backward pass to get analytical gradients */
    backward_pass(batch_inputs, batch_targets, nn, batch_size,
                  h_pre, h_act, o_pre, o_act);

    int failed = 0;

    /* Check hidden layer weights */
    for (int i = 0; i < hidden_size; i++) {
        for (int j = 0; j < input_size; j++) {
            int idx = i * input_size + j;
            char label[64];
            snprintf(label, sizeof(label), "W_hidden[%d][%d]", i, j);
            failed += check_weight_gradient(nn, batch_inputs, batch_targets,
                                             batch_size,
                                             &nn->hidden.weights[idx],
                                             nn->hidden.weight_grads[idx],
                                             label);
        }
    }

    /* Check hidden layer biases */
    for (int i = 0; i < hidden_size; i++) {
        char label[32];
        snprintf(label, sizeof(label), "b_hidden[%d]", i);
        failed += check_weight_gradient(nn, batch_inputs, batch_targets,
                                         batch_size,
                                         &nn->hidden.biases[i],
                                         nn->hidden.bias_grads[i],
                                         label);
    }

    /* Check output layer weights */
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < hidden_size; j++) {
            int idx = i * hidden_size + j;
            char label[64];
            snprintf(label, sizeof(label), "W_output[%d][%d]", i, j);
            failed += check_weight_gradient(nn, batch_inputs, batch_targets,
                                             batch_size,
                                             &nn->output.weights[idx],
                                             nn->output.weight_grads[idx],
                                             label);
        }
    }

    /* Check output layer biases */
    for (int i = 0; i < output_size; i++) {
        char label[32];
        snprintf(label, sizeof(label), "b_output[%d]", i);
        failed += check_weight_gradient(nn, batch_inputs, batch_targets,
                                         batch_size,
                                         &nn->output.biases[i],
                                         nn->output.bias_grads[i],
                                         label);
    }

    int total_params = (hidden_size * input_size) + hidden_size
                     + (output_size * hidden_size) + output_size;

    if (failed == 0)
        printf("  All %d gradients correct (batch_size=%d, net=%d-%d-%d)\n",
               total_params, batch_size, input_size, hidden_size, output_size);

    /* cleanup */
    nn_free(nn);
    free(batch_inputs); free(batch_targets);
    free(h_pre); free(h_act); free(o_pre); free(o_act);

    return failed;
}

/* ---- public entry point ---- */

int run_nn_gradient_tests(void) {
    int failed = 0;

    printf("\n=== Gradient Correctness (Finite Difference) ===\n");
    printf("epsilon=%.0e  abs_tol=%.0e  rel_tol=%.0e\n",
           (double)EPSILON, (double)GRAD_ABS_TOL, (double)GRAD_REL_TOL);

    /* Tiny network, single sample — 11 parameters */
    printf("\n--- Tiny network (3-2-1), batch_size=1 ---\n");
    failed += check_all_gradients(3, 2, 1, 1);

    /* Same network, batch_size=2 — gradient is average of per-sample gradients */
    printf("\n--- Tiny network (3-2-1), batch_size=2 ---\n");
    failed += check_all_gradients(3, 2, 1, 2);

    /* Slightly larger */
    printf("\n--- Small network (4-3-2), batch_size=1 ---\n");
    failed += check_all_gradients(4, 3, 2, 1);

    printf("\nGradient correctness: %d failures\n", failed);
    return failed;
}
