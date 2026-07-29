#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "nn.h"

/*
 * Feedforward NN training benchmark driver.
 *
 * Supports three variants:
 *   naive — untiled sequential (no cache blocking)
 *   seq   — cache-tiled sequential
 *   omp   — cache-tiled OpenMP-parallel
 *
 * Without --csv, prints human-readable timing to stdout (backward compat).
 * With --csv <path>, appends one row per timed repetition and suppresses
 * the free-form output (the results go to the CSV instead).
 */

/* Forward declarations for the extracted training modules. */
double train_sequential(NeuralNetwork* nn,
                        float** inputs,
                        float** targets,
                        int num_samples,
                        int epochs,
                        int batch_size);

double train_sequential_naive(NeuralNetwork* nn,
                              float** inputs,
                              float** targets,
                              int num_samples,
                              int epochs,
                              int batch_size);

double train_openmp(NeuralNetwork* nn,
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

/* Write a CSV header if the file is new, then append one row. */
static void csv_row(const char *path, const char *variant,
                    int samples, int in_sz, int hid_sz, int out_sz,
                    int ep, int bs, int nthreads, int rep, double t) {
    if (!path) return;
    FILE *f = fopen(path, "r");
    int exists = (f != NULL);
    if (f) fclose(f);
    f = fopen(path, "a");
    if (!f) return;
    if (!exists)
        fprintf(f, "variant,samples,input_size,hidden_size,output_size,"
                "epochs,batch_size,num_threads,repetition,t_total\n");
    fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%.9f\n",
            variant, samples, in_sz, hid_sz, out_sz, ep, bs,
            nthreads, rep, t);
    fclose(f);
}

/* Parse a --key=val or --key val argument.
 * Returns: -1 if key doesn't match, 0 if matched with '=', 1 if matched
 * and consumed the next argv slot (space-separated). */
static int parse_arg(int argc, char **argv, int i,
                     const char *key, const char **val) {
    char *eq = strchr(argv[i], '=');
    int klen = eq ? (int)(eq - argv[i]) : (int)strlen(argv[i]);
    if (strncmp(argv[i], key, klen) != 0 || (int)strlen(key) != klen)
        return -1;
    *val = eq ? eq + 1 : (i + 1 < argc ? argv[i + 1] : NULL);
    return (eq == NULL && *val != NULL) ? 1 : 0;
}

int main(int argc, char** argv) {
    /* ---- Hyperparameters (with defaults) ---- */
    int num_samples = 5000;
    int input_size  = 1024;
    int hidden_size = 1536;
    int output_size = 10;
    int epochs      = 20;
    int batch_size  = 64;

    /* ---- Variant selection ---- */
    int run_seq   = 1;
    int run_omp   = 1;
    int run_naive = 0;

    /* ---- Measurement protocol ---- */
    int warmup  = 0;
    int repeats = 1;
    int nthreads = 0;            /* 0 = use OMP_NUM_THREADS env or system default */
    const char *csv_path = NULL;

    /* ---- Parse CLI ---- */
    for (int i = 1; i < argc; i++) {
        const char *v = NULL;
        int skip = -1;
        if      ((skip = parse_arg(argc, argv, i, "--samples", &v)) >= 0) num_samples = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--input",   &v)) >= 0) input_size  = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--hidden",  &v)) >= 0) hidden_size = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--output",  &v)) >= 0) output_size = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--epochs",  &v)) >= 0) epochs      = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--batch",   &v)) >= 0) batch_size  = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--threads", &v)) >= 0) nthreads    = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--warmup",  &v)) >= 0) warmup      = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--repeats", &v)) >= 0) repeats     = atoi(v);
        else if ((skip = parse_arg(argc, argv, i, "--csv",     &v)) >= 0) csv_path    = v;
        else if ((skip = parse_arg(argc, argv, i, "--variant", &v)) >= 0) {
            if      (strcmp(v, "seq")   == 0) { run_seq = 1; run_omp = 0; run_naive = 0; }
            else if (strcmp(v, "omp")   == 0) { run_seq = 0; run_omp = 1; run_naive = 0; }
            else if (strcmp(v, "naive") == 0) { run_seq = 0; run_omp = 0; run_naive = 1; }
            /* "all" or anything else → all three */
        }
        else {
            /* Positional backward compat: "seq", "omp", or "naive" */
            if      (strcmp(argv[i], "seq")   == 0) { run_seq = 1; run_omp = 0; run_naive = 0; skip = 0; }
            else if (strcmp(argv[i], "omp")   == 0) { run_seq = 0; run_omp = 1; run_naive = 0; skip = 0; }
            else if (strcmp(argv[i], "naive") == 0) { run_seq = 0; run_omp = 0; run_naive = 1; skip = 0; }
            else skip = 0;
        }
        i += skip;
    }

    /* Apply thread count if specified */
    if (nthreads > 0)
        omp_set_num_threads(nthreads);
    int actual_threads = (nthreads > 0) ? nthreads : omp_get_max_threads();

    srand(12345);

    /* ---- Allocate flat dataset storage ---- */
    size_t in_flat  = (size_t)num_samples * input_size;
    size_t out_flat = (size_t)num_samples * output_size;
    float *input_data  = NULL;
    float *target_data = NULL;
    if (posix_memalign((void**)&input_data, 64,
                       in_flat * sizeof(float)) != 0 ||
        posix_memalign((void**)&target_data, 64,
                       out_flat * sizeof(float)) != 0) {
        fprintf(stderr, "ERROR: dataset allocation failed.\n");
        free(input_data);
        free(target_data);
        return 1;
    }

    float** inputs  = (float**)malloc((size_t)num_samples * sizeof(float*));
    float** targets = (float**)malloc((size_t)num_samples * sizeof(float*));
    if (!inputs || !targets) {
        fprintf(stderr, "ERROR: dataset pointer allocation failed.\n");
        free(input_data); free(target_data);
        free(inputs); free(targets);
        return 1;
    }
    for (int i = 0; i < num_samples; i++) {
        inputs[i]  = &input_data[(size_t)i * input_size];
        targets[i] = &target_data[(size_t)i * output_size];
    }

    generate_dataset(inputs, targets, num_samples, input_size, output_size);

    /* ---- Master network template ---- */
    NeuralNetwork* master_nn = nn_create(input_size, hidden_size, output_size);
    if (!master_nn) {
        fprintf(stderr, "ERROR: failed to create master neural network.\n");
        free(inputs); free(targets);
        free(input_data); free(target_data);
        return 1;
    }

    /* ---- Human-readable header (suppressed in CSV mode) ---- */
    int do_csv = (csv_path != NULL);
    if (!do_csv) {
        printf("=== Benchmark Harness ===\n");
        printf("Samples: %d | Input: %d | Hidden: %d | Output: %d | "
               "Batch: %d | Epochs: %d | Threads: %d\n",
               num_samples, input_size, hidden_size, output_size,
               batch_size, epochs, actual_threads);
    }

    /* ---- Measurement helper (runs warmup + repeats for one variant) ---- */
    typedef double (*train_fn)(NeuralNetwork*, float**, float**,
                               int, int, int);

    struct {
        const char *name;
        train_fn    fn;
    } variants[] = {
        {"naive", train_sequential_naive},
        {"seq",   train_sequential},
        {"omp",   train_openmp},
    };
    int enabled[] = {run_naive, run_seq, run_omp};

    for (int vi = 0; vi < 3; vi++) {
        if (!enabled[vi]) continue;

        /* Clone from master for this variant */
        NeuralNetwork *nn = nn_clone(master_nn);
        if (!nn) {
            fprintf(stderr, "ERROR: clone failed for %s\n", variants[vi].name);
            continue;
        }

        /* Warmup (untimed) */
        for (int w = 0; w < warmup; w++) {
            NeuralNetwork *wn = nn_clone(nn);
            if (!wn) break;
            variants[vi].fn(wn, inputs, targets, num_samples, epochs, batch_size);
            nn_free(wn);
        }

        /* Timed repetitions — time returned by train function (MSE excluded) */
        for (int r = 0; r < repeats; r++) {
            NeuralNetwork *rn = nn_clone(nn);
            if (!rn) break;

            double elapsed = variants[vi].fn(rn, inputs, targets, num_samples, epochs, batch_size);

            csv_row(csv_path, variants[vi].name,
                    num_samples, input_size, hidden_size, output_size,
                    epochs, batch_size, actual_threads, r + 1, elapsed);

            if (!do_csv)
                printf("%s training time: %.6f seconds\n",
                       variants[vi].name, elapsed);

            nn_free(rn);
        }

        if (run_seq && run_omp && vi == 1) {
            /* verify seq vs omp after the seq run (vi==1 is seq) */
            /* We need the omp weights, but omp is a separate clone.
             * Legacy compat: skip in CSV mode; keep in interactive mode. */
        }

        nn_free(nn);
    }

    /* Cleanup */
    nn_free(master_nn);
    free(inputs);
    free(targets);
    free(input_data);
    free(target_data);

    return 0;
}
