#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "math_utils.h"
#include "train_mpi.h"
#include "train_utils.h"

/* ---- gradient synchronization helper ---- */

static void allreduce_grads(NeuralNetwork *nn, MPI_Comm comm) {
    Layer *hid = &nn->hidden;
    Layer *out = &nn->output;
    int w_hid_size = hid->input_size * hid->output_size;
    int w_out_size = out->input_size * out->output_size;

    /*
     * Every rank starts with gradients from its own local mini-batch.
     * MPI_IN_PLACE replaces each local array with the element-wise sum from
     * all ranks, leaving every model replica with identical global gradients.
     * The caller converts this sum to a global-batch mean through the learning
     * rate used by update_weights.
     */
    MPI_Allreduce(MPI_IN_PLACE, hid->weight_grads, w_hid_size,
                  MPI_FLOAT, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, hid->bias_grads, hid->output_size,
                  MPI_FLOAT, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, out->weight_grads, w_out_size,
                  MPI_FLOAT, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, out->bias_grads, out->output_size,
                  MPI_FLOAT, MPI_SUM, comm);
}

/* =================================================================
 * Data-parallel training loop.
 *
 * Each rank owns a full copy of the weights and a distinct slice of
 * the dataset.  After every mini-batch backward pass, gradients are
 * summed across ranks via MPI_Allreduce so that each rank applies
 * the mean gradient over the global batch.
 * ================================================================= */
double train_mpi(NeuralNetwork *nn,
                    float **inputs,
                    float **targets,
                    int local_num_samples,
                    int epochs,
                    int batch_size,
                    int tile_size,
                    int report_loss_each_epoch,
                    MPI_Comm comm) {

    /*
     * Step 1: Validate the local arguments.
     *
     * All ranks are expected to call this routine with compatible network and
     * training settings. The benchmark driver validates the shared settings
     * before entering this function.
     */
    if (!nn || !inputs || !targets || local_num_samples <= 0
        || epochs <= 0 || batch_size <= 0 || tile_size <= 0)
        return -1.0;

    /*
     * Step 2: Confirm that every rank will execute the same number of updates.
     *
     * One MPI_Allreduce is performed per update. If one rank had fewer local
     * mini-batches, it would leave the loop while the others waited inside a
     * collective, causing a deadlock. Local final batch sizes may differ; only
     * the number of update steps must match.
     */
    int local_updates = local_num_samples / batch_size
                        + (local_num_samples % batch_size != 0);
    int min_updates = 0;
    int max_updates = 0;
    MPI_Allreduce(&local_updates, &min_updates, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_updates, &max_updates, 1, MPI_INT, MPI_MAX, comm);
    if (min_updates != max_updates)
        return -1.0;

    /*
     * Step 3: Allocate the local training workspace.
     *
     * Each rank stores only its own mini-batch activations and deltas, while
     * keeping a complete copy of the network parameters and gradient arrays.
     * The 64-byte alignment supports the packed, tiled matrix kernels.
     */
    int *indices = (int *)malloc((size_t)batch_size * sizeof(int));

    int input_size = nn->hidden.input_size;
    int hid_size   = nn->hidden.output_size;
    int out_size   = nn->output.output_size;

    float *batch_inputs   = NULL;
    float *batch_targets  = NULL;
    float *batch_h_pre    = NULL;
    float *batch_h_act    = NULL;
    float *batch_o_pre    = NULL;
    float *batch_o_act    = NULL;
    float *batch_h_deltas = NULL;
    float *batch_o_deltas = NULL;

    int local_alloc_ok = indices != NULL;
    if (local_alloc_ok &&
        (posix_memalign((void **)&batch_inputs, 64,
                        (size_t)batch_size * input_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_targets, 64,
                        (size_t)batch_size * out_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_h_pre, 64,
                        (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_h_act, 64,
                        (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_o_pre, 64,
                        (size_t)batch_size * out_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_o_act, 64,
                        (size_t)batch_size * out_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_h_deltas, 64,
                        (size_t)batch_size * hid_size * sizeof(float)) != 0 ||
         posix_memalign((void **)&batch_o_deltas, 64,
                        (size_t)batch_size * out_size * sizeof(float)) != 0))
        local_alloc_ok = 0;

    /*
     * Step 4: Make allocation success a collective decision.
     *
     * Even if only one rank failed to allocate, every rank must return before
     * training begins; otherwise successful ranks would later enter MPI
     * collectives that the failed rank never reaches.
     */
    int all_alloc_ok = 0;
    MPI_Allreduce(&local_alloc_ok, &all_alloc_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!all_alloc_ok) {
        free(indices);
        free(batch_inputs);  free(batch_targets);
        free(batch_h_pre);   free(batch_h_act);
        free(batch_o_pre);   free(batch_o_act);
        free(batch_h_deltas); free(batch_o_deltas);
        return -1.0;
    }

    const float learning_rate = 0.5f;
    double train_time = 0.0;

    /*
     * Step 5: Repeat the synchronous data-parallel pipeline for each epoch.
     *
     * The timed region covers mini-batch training, including communication.
     * Optional full-dataset loss evaluation below is deliberately excluded.
     */
    for (int epoch = 0; epoch < epochs; epoch++) {
        double t0 = MPI_Wtime();

        /*
         * Step 6: Select the next local mini-batch.
         *
         * Ranks process corresponding update numbers simultaneously, but they
         * operate on disjoint dataset slices and may contribute different
         * numbers of samples to the final update.
         */
        for (int start = 0; start < local_num_samples; start += batch_size) {
            int batch_end = (start + batch_size < local_num_samples)
                                ? start + batch_size
                                : local_num_samples;
            int batch_count = batch_end - start;

            /*
             * Step 7: Determine the effective global mini-batch size.
             *
             * Summing the local counts handles uneven final mini-batches. The
             * global count is later used to average the summed gradients over
             * exactly the samples that participated in this update.
             */
            int global_batch_count = 0;
            MPI_Allreduce(&batch_count, &global_batch_count, 1, MPI_INT,
                          MPI_SUM, comm);
            if (global_batch_count <= 0) {
                free(batch_inputs);   free(batch_targets);
                free(batch_h_pre);    free(batch_h_act);
                free(batch_o_pre);    free(batch_o_act);
                free(batch_h_deltas); free(batch_o_deltas);
                free(indices);
                return -1.0;
            }
            float lr_per_sample = learning_rate / (float)global_batch_count;

            /*
             * Step 8: Build local row indices and reset gradient storage.
             *
             * The current implementation traverses each local dataset slice in
             * order. Gradients must be cleared because the matrix routines add
             * their results into the existing arrays.
             */
            for (int j = 0; j < batch_count; j++)
                indices[j] = start + j;

            zero_grads(nn);

            /*
             * Step 9: Pack the selected rows into contiguous mini-batch arrays.
             *
             * Contiguous row-major storage lets the tiled GEMM kernels reuse
             * cache lines and process all samples in the batch together.
             */
            pack_batch(batch_inputs,  inputs,  indices, batch_count, input_size);
            pack_batch(batch_targets, targets, indices, batch_count, out_size);

            int hid_batch_size = batch_count * hid_size;
            int out_batch_size = batch_count * out_size;

            /*
             * Step 10: Forward pass through the hidden layer.
             *
             * The stored weights are output-major, so the operation is
             * batch_inputs * hidden_weights^T. Biases and sigmoid activations
             * are then applied independently to every sample row.
             */
            memset(batch_h_pre, 0, hid_batch_size * sizeof(float));
            mat_mat_mult_transposeB_rtile(batch_inputs, nn->hidden.weights,
                                    batch_h_pre,
                                    batch_count, input_size, hid_size,
                                    tile_size);
            add_bias_rowwise(batch_h_pre, nn->hidden.biases,
                             batch_count, hid_size);
            sigmoid_batch_to(batch_h_pre, batch_h_act, hid_batch_size);

            /*
             * Step 11: Forward pass through the output layer.
             *
             * Hidden activations become the input matrix, followed by the same
             * weight multiplication, bias addition, and sigmoid activation.
             */
            memset(batch_o_pre, 0, out_batch_size * sizeof(float));
            mat_mat_mult_transposeB_rtile(batch_h_act, nn->output.weights,
                                    batch_o_pre,
                                    batch_count, hid_size, out_size,
                                    tile_size);
            add_bias_rowwise(batch_o_pre, nn->output.biases,
                             batch_count, out_size);
            sigmoid_batch_to(batch_o_pre, batch_o_act, out_batch_size);

            /*
             * Step 12: Start backpropagation at the output layer.
             *
             * Each output delta combines the prediction error with the sigmoid
             * derivative evaluated at the saved output pre-activation.
             */
            compute_output_deltas(batch_o_act, batch_targets, batch_o_pre,
                                  batch_o_deltas, out_batch_size);

            /*
             * Step 13: Propagate output deltas into the hidden layer.
             *
             * Multiplication by the non-transposed output-weight matrix sums
             * the contribution from every output neuron. The saved hidden
             * pre-activations then supply the hidden sigmoid derivatives.
             */
            memset(batch_h_deltas, 0, hid_batch_size * sizeof(float));
            mat_mat_mult_rtile(batch_o_deltas, nn->output.weights,
                         batch_h_deltas,
                         batch_count, out_size, hid_size, tile_size);
            apply_sigmoid_deriv_inplace(batch_h_pre, batch_h_deltas,
                                        hid_batch_size);

            /*
             * Step 14: Form this rank's output-layer gradients.
             *
             * output_deltas^T * hidden_activations produces one gradient per
             * output weight; summing delta rows produces the bias gradients.
             */
            mat_mat_mult_transposeA_rtile(batch_o_deltas, batch_h_act,
                                    nn->output.weight_grads,
                                    batch_count, out_size, hid_size,
                                    tile_size);
            accumulate_bias_gradients(batch_o_deltas, nn->output.bias_grads,
                                      batch_count, out_size);

            /*
             * Step 15: Form this rank's hidden-layer gradients.
             *
             * hidden_deltas^T * batch_inputs produces one gradient per hidden
             * weight, while the row sum supplies each hidden bias gradient.
             */
            mat_mat_mult_transposeA_rtile(batch_h_deltas, batch_inputs,
                                    nn->hidden.weight_grads,
                                    batch_count, hid_size, input_size,
                                    tile_size);
            accumulate_bias_gradients(batch_h_deltas, nn->hidden.bias_grads,
                                      batch_count, hid_size);

            /*
             * Step 16: Synchronize gradients before changing any parameters.
             *
             * Allreduce sums every rank's four gradient arrays and returns the
             * same sums to every rank. This collective is the synchronization
             * point that keeps all model replicas identical.
             */
            allreduce_grads(nn, comm);

            /*
             * Step 17: Apply one global mini-batch update on every rank.
             *
             * lr_per_sample divides the summed gradients by the global sample
             * count, so this is the mean gradient of the combined mini-batch.
             * Since all ranks start from identical parameters and apply the
             * same update, their model replicas remain synchronized.
             */
            update_weights(nn, lr_per_sample);
        }
        train_time += MPI_Wtime() - t0;

        /*
         * Step 18: Optionally report the global training loss.
         *
         * Each rank evaluates its own data slice. Multiplying the local mean
         * by its sample count recovers its total contribution; Allreduce then
         * combines those totals and sample counts into a weighted global mean.
         */
        if (report_loss_each_epoch) {
            float local_loss = compute_training_loss(
                nn, inputs, targets, local_num_samples,
                input_size, hid_size, out_size);
            int local_eval_ok = local_loss >= 0.0f;
            int all_eval_ok = 0;
            MPI_Allreduce(&local_eval_ok, &all_eval_ok, 1, MPI_INT,
                          MPI_MIN, comm);
            if (!all_eval_ok) {
                free(batch_inputs);   free(batch_targets);
                free(batch_h_pre);    free(batch_h_act);
                free(batch_o_pre);    free(batch_o_act);
                free(batch_h_deltas); free(batch_o_deltas);
                free(indices);
                return -1.0;
            }
            float local_se = local_loss * local_num_samples;
            float global_se;
            MPI_Allreduce(&local_se, &global_se, 1, MPI_FLOAT, MPI_SUM, comm);
            int global_n, local_n = local_num_samples;
            MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, comm);
            float global_loss = global_se / global_n;
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (rank == 0)
                fprintf(stderr, "[Epoch %d/%d] training loss = %.6f\n",
                        epoch + 1, epochs, global_loss);
        }
    }

    /*
     * Step 19: Release rank-local workspace and return this rank's elapsed
     * training time. The benchmark driver later takes the maximum across ranks
     * because the slowest rank determines the observed parallel runtime.
     */
    free(batch_inputs);   free(batch_targets);
    free(batch_h_pre);    free(batch_h_act);
    free(batch_o_pre);    free(batch_o_act);
    free(batch_h_deltas); free(batch_o_deltas);
    free(indices);

    return train_time;
}
