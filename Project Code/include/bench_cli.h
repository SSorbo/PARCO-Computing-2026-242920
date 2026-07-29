#ifndef BENCH_CLI_H
#define BENCH_CLI_H

#include <stddef.h>
#include <stdio.h>

typedef struct {
    int samples;
    int input;
    int hidden;
    int output;
    int epochs;
    int batch;
    int repeats;
    int warmup;
    int no_header;
    int append;
    int tile;
    int max_threads;
    int report_loss_each_epoch;
    const char *csv_path;
    const char *experiment;
    const char *variant;
    const char *thread_list;
} BenchConfig;

typedef struct {
    const char *experiment;
    const char *backend;
    const char *variant;
    int samples;
    int input_size;
    int hidden_size;
    int output_size;
    int epochs;
    int batch_size;
    int num_threads;
    int num_ranks;
    int tile_size;
    int repetition;
    int warmup_reps;
    double t_train;
    double t_loss_eval;
    double t_total;
    float final_loss;
    int updates_per_epoch;
    int global_batch_size;
    int last_global_batch_size;
} BenchResult;

void bench_config_defaults(BenchConfig *cfg);
int bench_parse_common_arg(BenchConfig *cfg, int argc, char **argv, int *i);
int bench_parse_int_list(const char *s, int *out, int max_count);
int bench_validate_config(const BenchConfig *cfg, int require_threads,
                          int require_tile, char *error, size_t error_size);
FILE *bench_open_csv(const BenchConfig *cfg);
double bench_wtime(void);
int bench_ceil_div(int a, int b);
void bench_write_csv_header(FILE *f);
void bench_write_csv_row(FILE *f, const BenchResult *row);

#endif
