#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"
#include "train.h"
#include "train_mpi.h"

typedef struct {
    float *input_data;
    float *target_data;
    float **inputs;
    float **targets;
    int samples;
} TestDataset;

static int dataset_init(TestDataset *ds, int rank, int samples,
                        int input_size, int output_size) {
    memset(ds, 0, sizeof(*ds));
    ds->samples = samples;
    if (posix_memalign((void **)&ds->input_data, 64,
                       (size_t)samples * input_size * sizeof(float)) != 0 ||
        posix_memalign((void **)&ds->target_data, 64,
                       (size_t)samples * output_size * sizeof(float)) != 0) {
        free(ds->input_data);
        free(ds->target_data);
        return -1;
    }

    ds->inputs = malloc((size_t)samples * sizeof(float *));
    ds->targets = malloc((size_t)samples * sizeof(float *));
    if (!ds->inputs || !ds->targets) {
        free(ds->input_data);
        free(ds->target_data);
        free(ds->inputs);
        free(ds->targets);
        return -1;
    }

    for (int s = 0; s < samples; s++) {
        ds->inputs[s] = ds->input_data + (size_t)s * input_size;
        ds->targets[s] = ds->target_data + (size_t)s * output_size;
        for (int i = 0; i < input_size; i++) {
            int value = ((rank + 1) * 11 + (s + 1) * 7 + (i + 1) * 3) % 29;
            ds->inputs[s][i] = (float)value / 29.0f;
        }
        int label = 0;
        for (int i = 1; i < output_size; i++) {
            if (ds->inputs[s][i] > ds->inputs[s][label])
                label = i;
        }
        for (int i = 0; i < output_size; i++)
            ds->targets[s][i] = i == label ? 1.0f : 0.0f;
    }
    return 0;
}

static void dataset_free(TestDataset *ds) {
    free(ds->input_data);
    free(ds->target_data);
    free(ds->inputs);
    free(ds->targets);
}

static size_t parameter_count(const NeuralNetwork *nn) {
    return (size_t)nn->hidden.input_size * nn->hidden.output_size
         + (size_t)nn->hidden.output_size
         + (size_t)nn->output.input_size * nn->output.output_size
         + (size_t)nn->output.output_size;
}

static void pack_parameters(const NeuralNetwork *nn, float *packed) {
    size_t offset = 0;
    size_t count = (size_t)nn->hidden.input_size * nn->hidden.output_size;
    memcpy(packed + offset, nn->hidden.weights, count * sizeof(float));
    offset += count;
    count = (size_t)nn->hidden.output_size;
    memcpy(packed + offset, nn->hidden.biases, count * sizeof(float));
    offset += count;
    count = (size_t)nn->output.input_size * nn->output.output_size;
    memcpy(packed + offset, nn->output.weights, count * sizeof(float));
    offset += count;
    count = (size_t)nn->output.output_size;
    memcpy(packed + offset, nn->output.biases, count * sizeof(float));
}

static float max_parameter_diff(const NeuralNetwork *a,
                                const NeuralNetwork *b) {
    size_t count = parameter_count(a);
    float *pa = malloc(count * sizeof(float));
    float *pb = malloc(count * sizeof(float));
    if (!pa || !pb) {
        free(pa);
        free(pb);
        return INFINITY;
    }
    pack_parameters(a, pa);
    pack_parameters(b, pb);
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(pa[i] - pb[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    free(pa);
    free(pb);
    return max_diff;
}

static int test_one_rank_equivalence(int rank, int size) {
    if (size != 1)
        return 0;

    TestDataset ds;
    if (dataset_init(&ds, rank, 7, 4, 2) != 0)
        return 1;

    srand(12345);
    NeuralNetwork *master = nn_create(4, 3, 2);
    NeuralNetwork *seq = nn_clone(master);
    NeuralNetwork *mpi = nn_clone(master);
    nn_free(master);
    if (!seq || !mpi) {
        nn_free(seq);
        nn_free(mpi);
        dataset_free(&ds);
        return 1;
    }

    double seq_time = train_sequential(seq, ds.inputs, ds.targets,
                                          ds.samples, 2, 3, 0);
    double mpi_time = train_mpi(mpi, ds.inputs, ds.targets, ds.samples,
                                   2, 3, 64, 0, MPI_COMM_WORLD);
    if (seq_time < 0.0 || mpi_time < 0.0) {
        nn_free(seq);
        nn_free(mpi);
        dataset_free(&ds);
        return 1;
    }
    float diff = max_parameter_diff(seq, mpi);
    printf("MPI 1-rank equivalence max parameter diff: %.3e\n", diff);

    nn_free(seq);
    nn_free(mpi);
    dataset_free(&ds);
    return diff > 1e-6f;
}

static int test_uneven_batch_replica_sync(int rank, int size) {
    const int input_size = 4;
    const int hidden_size = 3;
    const int output_size = 2;
    const int batch_size = 3;
    int local_samples = rank == 0 ? 5 : 4;

    TestDataset ds;
    if (dataset_init(&ds, rank, local_samples, input_size, output_size) != 0)
        return 1;

    srand(54321);
    NeuralNetwork *nn = nn_create(input_size, hidden_size, output_size);
    if (!nn) {
        dataset_free(&ds);
        return 1;
    }

    if (train_mpi(nn, ds.inputs, ds.targets, local_samples, 2, batch_size,
                     64, 0, MPI_COMM_WORLD) < 0.0) {
        nn_free(nn);
        dataset_free(&ds);
        return 1;
    }

    size_t count = parameter_count(nn);
    float *local = malloc(count * sizeof(float));
    float *reference = malloc(count * sizeof(float));
    int allocation_ok = local != NULL && reference != NULL;
    int all_allocation_ok = 0;
    MPI_Allreduce(&allocation_ok, &all_allocation_ok, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if (!all_allocation_ok) {
        free(local);
        free(reference);
        nn_free(nn);
        dataset_free(&ds);
        return 1;
    }

    pack_parameters(nn, local);
    if (rank == 0)
        memcpy(reference, local, count * sizeof(float));
    MPI_Bcast(reference, (int)count, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float local_max = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = fabsf(local[i] - reference[i]);
        if (diff > local_max)
            local_max = diff;
    }
    float global_max = 0.0f;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (rank == 0)
        printf("Uneven-batch replica max parameter diff (%d ranks): %.3e\n",
               size, global_max);

    free(local);
    free(reference);
    nn_free(nn);
    dataset_free(&ds);
    return global_max != 0.0f;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int failed = 0;
    failed += test_one_rank_equivalence(rank, size);
    failed += test_uneven_batch_replica_sync(rank, size);

    int global_failed = 0;
    MPI_Allreduce(&failed, &global_failed, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (rank == 0)
        printf("MPI correctness tests: %s\n",
               global_failed == 0 ? "PASS" : "FAIL");

    MPI_Finalize();
    return global_failed == 0 ? 0 : 1;
}
