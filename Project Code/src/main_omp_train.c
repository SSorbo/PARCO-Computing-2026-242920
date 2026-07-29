#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "bench_cli.h"
#include "nn.h"
#include "train.h"
#include "train_utils.h"

static void generate_dataset(float **inputs, float **targets,
                             int num_samples, int input_sz, int output_sz) {
    for (int s = 0; s < num_samples; s++) {
        for (int f = 0; f < input_sz; f++)
            inputs[s][f] = (float)rand() / (float)RAND_MAX;

        int label = 0;
        float best = inputs[s][0];
        for (int k = 1; k < output_sz; k++) {
            if (inputs[s][k] > best) {
                best = inputs[s][k];
                label = k;
            }
        }

        for (int k = 0; k < output_sz; k++)
            targets[s][k] = (k == label) ? 1.0f : 0.0f;
    }
}

typedef struct {
    float *input_data;
    float *target_data;
    float **inputs;
    float **targets;
} Dataset;

static int dataset_init(Dataset *ds, int samples, int input_size, int output_size) {
    memset(ds, 0, sizeof(*ds));
    size_t input_count = (size_t)samples * input_size;
    size_t target_count = (size_t)samples * output_size;

    if (posix_memalign((void **)&ds->input_data, 64,
                       input_count * sizeof(float)) != 0 ||
        posix_memalign((void **)&ds->target_data, 64,
                       target_count * sizeof(float)) != 0) {
        free(ds->input_data);
        free(ds->target_data);
        ds->input_data = NULL;
        ds->target_data = NULL;
        return -1;
    }

    ds->inputs = malloc((size_t)samples * sizeof(float *));
    ds->targets = malloc((size_t)samples * sizeof(float *));
    if (!ds->inputs || !ds->targets) {
        free(ds->input_data);
        free(ds->target_data);
        free(ds->inputs);
        free(ds->targets);
        memset(ds, 0, sizeof(*ds));
        return -1;
    }

    for (int i = 0; i < samples; i++) {
        ds->inputs[i] = &ds->input_data[(size_t)i * input_size];
        ds->targets[i] = &ds->target_data[(size_t)i * output_size];
    }

    generate_dataset(ds->inputs, ds->targets, samples, input_size, output_size);
    return 0;
}

static void dataset_free(Dataset *ds) {
    free(ds->input_data);
    free(ds->target_data);
    free(ds->inputs);
    free(ds->targets);
}

static int build_thread_list(const BenchConfig *cfg, int *thread_list,
                             int max_count) {
    int count = 0;

    if (cfg->thread_list) {
        count = bench_parse_int_list(cfg->thread_list, thread_list, max_count);
        return count;
    }

    for (int t = 1; t <= cfg->max_threads && count < max_count; ) {
        thread_list[count++] = t;
        if (t > cfg->max_threads / 2)
            break;
        t *= 2;
    }

    if (count == 0 || thread_list[count - 1] != cfg->max_threads) {
        if (count < max_count)
            thread_list[count++] = cfg->max_threads;
    }

    return count;
}

static int run_threads(FILE *csv, const BenchConfig *cfg,
                       NeuralNetwork *master, Dataset *ds,
                       int num_threads) {
    omp_set_num_threads(num_threads);

    for (int w = 0; w < cfg->warmup; w++) {
        NeuralNetwork *wn = nn_clone(master);
        if (!wn)
            return -1;
        double warmup_time = train_openmp(wn, ds->inputs, ds->targets,
                                             cfg->samples, cfg->epochs,
                                             cfg->batch,
                                             cfg->report_loss_each_epoch);
        nn_free(wn);
        if (warmup_time < 0.0)
            return -1;
    }

    for (int r = 0; r < cfg->repeats; r++) {
        NeuralNetwork *rn = nn_clone(master);
        if (!rn)
            return -1;

        double t_train = train_openmp(rn, ds->inputs, ds->targets,
                                         cfg->samples, cfg->epochs,
                                         cfg->batch,
                                         cfg->report_loss_each_epoch);
        if (t_train < 0.0) {
            nn_free(rn);
            return -1;
        }

        double v0 = bench_wtime();
        float final_loss = compute_training_loss(
            rn, ds->inputs, ds->targets, cfg->samples,
            cfg->input, cfg->hidden, cfg->output);
        double t_loss_eval = bench_wtime() - v0;
        if (final_loss < 0.0f) {
            nn_free(rn);
            return -1;
        }

        BenchResult row = {
            .experiment = cfg->experiment,
            .backend = "omp",
            .variant = "omp",
            .samples = cfg->samples,
            .input_size = cfg->input,
            .hidden_size = cfg->hidden,
            .output_size = cfg->output,
            .epochs = cfg->epochs,
            .batch_size = cfg->batch,
            .num_threads = num_threads,
            .num_ranks = 1,
            .tile_size = 0,
            .repetition = r + 1,
            .warmup_reps = cfg->warmup,
            .t_train = t_train,
            .t_loss_eval = t_loss_eval,
            .t_total = t_train + t_loss_eval,
            .final_loss = final_loss,
            .updates_per_epoch = bench_ceil_div(cfg->samples, cfg->batch),
            .global_batch_size = cfg->samples < cfg->batch
                                     ? cfg->samples
                                     : cfg->batch,
            .last_global_batch_size =
                cfg->samples - (bench_ceil_div(cfg->samples, cfg->batch) - 1) * cfg->batch,
        };
        bench_write_csv_row(csv, &row);
        fflush(csv);
        nn_free(rn);
    }
    return 0;
}

int main(int argc, char **argv) {
    BenchConfig cfg;
    bench_config_defaults(&cfg);
    cfg.variant = "omp";

    for (int i = 1; i < argc; i++) {
        int parsed = bench_parse_common_arg(&cfg, argc, argv, &i);
        if (parsed < 0) {
            fprintf(stderr, "Invalid or missing value for argument: %s\n", argv[i]);
            return 1;
        }
        if (parsed == 0) {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    char config_error[160];
    if (bench_validate_config(&cfg, 1, 0, config_error,
                              sizeof(config_error)) != 0) {
        fprintf(stderr, "Invalid benchmark configuration: %s\n", config_error);
        return 1;
    }
    if (strcmp(cfg.variant, "omp") != 0) {
        fprintf(stderr, "Invalid OpenMP variant: %s\n", cfg.variant);
        return 1;
    }

    int thread_list[64];
    int thread_count = build_thread_list(&cfg, thread_list, 64);
    if (thread_count <= 0) {
        fprintf(stderr, "FATAL: no valid OpenMP thread counts requested\n");
        return 1;
    }
    omp_set_dynamic(0);

    FILE *csv = bench_open_csv(&cfg);
    if (!csv) {
        fprintf(stderr, "Cannot open CSV: %s\n", cfg.csv_path);
        return 1;
    }
    if (!cfg.no_header)
        bench_write_csv_header(csv);

    srand(12345);

    fprintf(stderr, "=== OpenMP NN Training Benchmark ===\n");
    fprintf(stderr, "Experiment=%s Repeats=%d Warmup=%d\n",
            cfg.experiment, cfg.repeats, cfg.warmup);
    fprintf(stderr, "Samples=%d Network=%d->%d->%d Epochs=%d Batch=%d\n",
            cfg.samples, cfg.input, cfg.hidden, cfg.output,
            cfg.epochs, cfg.batch);
    fprintf(stderr, "ReportLossEachEpoch=%d ThreadCounts=",
            cfg.report_loss_each_epoch);
    for (int i = 0; i < thread_count; i++)
        fprintf(stderr, "%s%d", i ? "," : "", thread_list[i]);
    fprintf(stderr, "\n\n");

    Dataset ds;
    if (dataset_init(&ds, cfg.samples, cfg.input, cfg.output) != 0) {
        fprintf(stderr, "FATAL: dataset allocation failed\n");
        if (cfg.csv_path)
            fclose(csv);
        return 1;
    }

    /* Keep dataset and model RNG streams independent and backend-identical. */
    srand(12345);
    NeuralNetwork *master = nn_create(cfg.input, cfg.hidden, cfg.output);
    if (!master) {
        fprintf(stderr, "FATAL: nn_create failed\n");
        dataset_free(&ds);
        if (cfg.csv_path)
            fclose(csv);
        return 1;
    }

    int run_status = 0;
    for (int i = 0; i < thread_count; i++) {
        fprintf(stderr, "--- threads=%d ---\n", thread_list[i]);
        if (run_threads(csv, &cfg, master, &ds, thread_list[i]) != 0) {
            run_status = 1;
            break;
        }
    }

    nn_free(master);
    dataset_free(&ds);

    if (cfg.csv_path)
        fclose(csv);

    if (run_status != 0)
        fprintf(stderr, "FATAL: training workspace allocation failed\n");
    else
        fprintf(stderr, "Done.\n");
    return run_status;
}
