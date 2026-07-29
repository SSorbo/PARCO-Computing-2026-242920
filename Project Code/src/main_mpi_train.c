#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <mpi.h>

#include "bench_cli.h"
#include "nn.h"
#include "train_mpi.h"
#include "train_utils.h"

static int local_count(int total, int num_ranks, int rank) {
    int base = total / num_ranks;
    int rem = total % num_ranks;
    return base + (rank < rem ? 1 : 0);
}

static int local_offset(int total, int num_ranks, int rank) {
    int base = total / num_ranks;
    int rem = total % num_ranks;
    if (rank < rem)
        return rank * (base + 1);
    return rem * (base + 1) + (rank - rem) * base;
}

static void generate_dataset(float *input_data, float *target_data,
                             int num_samples, int input_size, int output_size) {
    for (int s = 0; s < num_samples; s++) {
        size_t input_off = (size_t)s * input_size;
        size_t target_off = (size_t)s * output_size;

        for (int f = 0; f < input_size; f++)
            input_data[input_off + f] = (float)rand() / (float)RAND_MAX;

        int label = 0;
        float best = input_data[input_off];
        for (int k = 1; k < output_size; k++) {
            if (input_data[input_off + k] > best) {
                best = input_data[input_off + k];
                label = k;
            }
        }

        for (int k = 0; k < output_size; k++)
            target_data[target_off + k] = (k == label) ? 1.0f : 0.0f;
    }
}

typedef struct {
    float *input_data;
    float *target_data;
    float **inputs;
    float **targets;
    int samples;
} LocalDataset;

static int local_dataset_init(LocalDataset *ds, int samples, int input_size,
                              int output_size, float *input_buf,
                              float *target_buf) {
    memset(ds, 0, sizeof(*ds));
    ds->samples = samples;
    ds->input_data = input_buf;
    ds->target_data = target_buf;
    ds->inputs = malloc((size_t)samples * sizeof(float *));
    ds->targets = malloc((size_t)samples * sizeof(float *));
    if (!ds->inputs || !ds->targets)
        return -1;

    for (int i = 0; i < samples; i++) {
        ds->inputs[i] = &ds->input_data[(size_t)i * input_size];
        ds->targets[i] = &ds->target_data[(size_t)i * output_size];
    }
    return 0;
}

static void local_dataset_free(LocalDataset *ds) {
    free(ds->input_data);
    free(ds->target_data);
    free(ds->inputs);
    free(ds->targets);
}

static void abort_all(const char *msg, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
        fprintf(stderr, "FATAL: %s\n", msg);
    MPI_Abort(comm, 1);
}

static void validate_update_counts(int local_n, int batch_size,
                                   MPI_Comm comm) {
    int local_updates = bench_ceil_div(local_n, batch_size);
    int min_updates = 0;
    int max_updates = 0;
    MPI_Allreduce(&local_updates, &min_updates, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(&local_updates, &max_updates, 1, MPI_INT, MPI_MAX, comm);
    if (min_updates != max_updates) {
        abort_all("MPI ranks would execute different mini-batch counts; "
                  "choose a sample/rank/batch configuration with equal counts.",
                  comm);
    }
}

static float compute_global_loss_timed(NeuralNetwork *nn, LocalDataset *ds,
                                       const BenchConfig *cfg, MPI_Comm comm,
                                       double *local_t_loss_eval) {
    double t0 = MPI_Wtime();
    float local_loss = compute_training_loss(
        nn, ds->inputs, ds->targets, ds->samples,
        cfg->input, cfg->hidden, cfg->output);
    int local_ok = local_loss >= 0.0f;
    int all_ok = 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!all_ok) {
        *local_t_loss_eval = MPI_Wtime() - t0;
        return -1.0f;
    }
    float local_se = local_loss * ds->samples;
    float global_se = 0.0f;
    int global_n = 0;
    int local_n = ds->samples;

    MPI_Allreduce(&local_se, &global_se, 1, MPI_FLOAT, MPI_SUM, comm);
    MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, comm);
    *local_t_loss_eval = MPI_Wtime() - t0;

    return global_n > 0 ? global_se / global_n : 0.0f;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int num_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    BenchConfig cfg;
    bench_config_defaults(&cfg);
    cfg.variant = "mpi";

    for (int i = 1; i < argc; i++) {
        int parsed = bench_parse_common_arg(&cfg, argc, argv, &i);
        if (parsed < 0)
            abort_all("invalid or missing MPI benchmark argument", MPI_COMM_WORLD);
        if (parsed == 0)
            abort_all("unknown MPI benchmark argument", MPI_COMM_WORLD);
    }

    char config_error[160];
    if (bench_validate_config(&cfg, 0, 1, config_error,
                              sizeof(config_error)) != 0)
        abort_all(config_error, MPI_COMM_WORLD);
    if (strcmp(cfg.variant, "mpi") != 0)
        abort_all("MPI variant must be 'mpi'", MPI_COMM_WORLD);
    if ((size_t)cfg.samples > (size_t)INT_MAX / (size_t)cfg.input ||
        (size_t)cfg.samples > (size_t)INT_MAX / (size_t)cfg.output)
        abort_all("dataset is too large for MPI_Scatterv integer counts",
                  MPI_COMM_WORLD);

    int local_n = local_count(cfg.samples, num_ranks, rank);
    validate_update_counts(local_n, cfg.batch, MPI_COMM_WORLD);
    int local_max_batch = local_n < cfg.batch ? local_n : cfg.batch;
    int global_max_batch = 0;
    MPI_Allreduce(&local_max_batch, &global_max_batch, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    int local_updates = bench_ceil_div(local_n, cfg.batch);
    int local_last_batch = local_n - (local_updates - 1) * cfg.batch;
    int global_last_batch = 0;
    MPI_Allreduce(&local_last_batch, &global_last_batch, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(stderr, "=== MPI NN Training Benchmark ===\n");
        fprintf(stderr, "Experiment=%s Ranks=%d Repeats=%d Warmup=%d\n",
                cfg.experiment, num_ranks, cfg.repeats, cfg.warmup);
        fprintf(stderr, "Samples=%d Network=%d->%d->%d Epochs=%d Batch=%d Tile=%d\n",
                cfg.samples, cfg.input, cfg.hidden, cfg.output,
                cfg.epochs, cfg.batch, cfg.tile);
        fprintf(stderr, "ReportLossEachEpoch=%d\n\n",
                cfg.report_loss_each_epoch);
    }

    float *recv_input = malloc((size_t)local_n * cfg.input * sizeof(float));
    float *recv_target = malloc((size_t)local_n * cfg.output * sizeof(float));
    if (!recv_input || !recv_target)
        abort_all("local dataset allocation failed", MPI_COMM_WORLD);

    if (rank == 0) {
        srand(12345);
        size_t total_input = (size_t)cfg.samples * cfg.input;
        size_t total_target = (size_t)cfg.samples * cfg.output;
        float *full_input = malloc(total_input * sizeof(float));
        float *full_target = malloc(total_target * sizeof(float));
        if (!full_input || !full_target)
            abort_all("rank 0 dataset allocation failed", MPI_COMM_WORLD);

        generate_dataset(full_input, full_target, cfg.samples,
                         cfg.input, cfg.output);

        int *sendcounts_in = malloc((size_t)num_ranks * sizeof(int));
        int *displs_in = malloc((size_t)num_ranks * sizeof(int));
        int *sendcounts_out = malloc((size_t)num_ranks * sizeof(int));
        int *displs_out = malloc((size_t)num_ranks * sizeof(int));
        if (!sendcounts_in || !displs_in || !sendcounts_out || !displs_out)
            abort_all("scatter metadata allocation failed", MPI_COMM_WORLD);

        for (int r = 0; r < num_ranks; r++) {
            int n_r = local_count(cfg.samples, num_ranks, r);
            int off_r = local_offset(cfg.samples, num_ranks, r);
            sendcounts_in[r] = n_r * cfg.input;
            displs_in[r] = off_r * cfg.input;
            sendcounts_out[r] = n_r * cfg.output;
            displs_out[r] = off_r * cfg.output;
        }

        MPI_Scatterv(full_input, sendcounts_in, displs_in, MPI_FLOAT,
                     recv_input, local_n * cfg.input, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(full_target, sendcounts_out, displs_out, MPI_FLOAT,
                     recv_target, local_n * cfg.output, MPI_FLOAT,
                     0, MPI_COMM_WORLD);

        free(sendcounts_in);
        free(displs_in);
        free(sendcounts_out);
        free(displs_out);
        free(full_input);
        free(full_target);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT,
                     recv_input, local_n * cfg.input, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT,
                     recv_target, local_n * cfg.output, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
    }

    LocalDataset ds;
    if (local_dataset_init(&ds, local_n, cfg.input, cfg.output,
                           recv_input, recv_target) != 0)
        abort_all("local dataset pointer setup failed", MPI_COMM_WORLD);

    srand(12345);
    NeuralNetwork *master = nn_create(cfg.input, cfg.hidden, cfg.output);
    if (!master)
        abort_all("nn_create failed", MPI_COMM_WORLD);

    FILE *csv = NULL;
    if (rank == 0) {
        csv = bench_open_csv(&cfg);
        if (!csv)
            abort_all("cannot open CSV", MPI_COMM_WORLD);
        if (!cfg.no_header)
            bench_write_csv_header(csv);
    }

    for (int w = 0; w < cfg.warmup; w++) {
        NeuralNetwork *wn = nn_clone(master);
        if (!wn)
            abort_all("MPI warmup model clone failed", MPI_COMM_WORLD);
        double warmup_time = train_mpi(wn, ds.inputs, ds.targets, local_n,
                                          cfg.epochs, cfg.batch, cfg.tile,
                                          cfg.report_loss_each_epoch,
                                          MPI_COMM_WORLD);
        if (warmup_time < 0.0)
            abort_all("MPI training workspace allocation failed",
                      MPI_COMM_WORLD);
        nn_free(wn);
    }

    for (int r = 0; r < cfg.repeats; r++) {
        NeuralNetwork *rn = nn_clone(master);
        if (!rn)
            abort_all("MPI timed model clone failed", MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double local_t_train = train_mpi(rn, ds.inputs, ds.targets,
                                            local_n, cfg.epochs,
                                            cfg.batch, cfg.tile,
                                            cfg.report_loss_each_epoch,
                                            MPI_COMM_WORLD);
        if (local_t_train < 0.0)
            abort_all("MPI training workspace allocation failed",
                      MPI_COMM_WORLD);
        double t_train = 0.0;
        MPI_Reduce(&local_t_train, &t_train, 1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        double local_t_loss_eval = 0.0;
        float final_loss = compute_global_loss_timed(
            rn, &ds, &cfg, MPI_COMM_WORLD, &local_t_loss_eval);
        if (final_loss < 0.0f)
            abort_all("MPI loss-evaluation workspace allocation failed",
                      MPI_COMM_WORLD);
        double t_loss_eval = 0.0;
        MPI_Reduce(&local_t_loss_eval, &t_loss_eval,
                   1, MPI_DOUBLE, MPI_MAX,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            BenchResult row = {
                .experiment = cfg.experiment,
                .backend = "mpi",
                .variant = "mpi",
                .samples = cfg.samples,
                .input_size = cfg.input,
                .hidden_size = cfg.hidden,
                .output_size = cfg.output,
                .epochs = cfg.epochs,
                .batch_size = cfg.batch,
                .num_threads = 1,
                .num_ranks = num_ranks,
                .tile_size = cfg.tile,
                .repetition = r + 1,
                .warmup_reps = cfg.warmup,
                .t_train = t_train,
                .t_loss_eval = t_loss_eval,
                .t_total = t_train + t_loss_eval,
                .final_loss = final_loss,
                .updates_per_epoch = local_updates,
                .global_batch_size = global_max_batch,
                .last_global_batch_size = global_last_batch,
            };
            bench_write_csv_row(csv, &row);
            fflush(csv);
        }

        nn_free(rn);
    }

    nn_free(master);
    local_dataset_free(&ds);

    if (rank == 0 && cfg.csv_path)
        fclose(csv);

    MPI_Finalize();
    return 0;
}
