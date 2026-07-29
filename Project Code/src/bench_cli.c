#include "bench_cli.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void bench_config_defaults(BenchConfig *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->samples = 5000;
    cfg->input = 1024;
    cfg->hidden = 1536;
    cfg->output = 10;
    cfg->epochs = 20;
    cfg->batch = 64;
    cfg->repeats = 3;
    cfg->warmup = 1;
    cfg->tile = 64;
    cfg->max_threads = 8;
    cfg->report_loss_each_epoch = 1;
    cfg->experiment = "default";
    cfg->variant = "both";
}

static int arg_value(int argc, char **argv, int i, const char *key,
                     const char **value, int *consumed) {
    size_t key_len = strlen(key);
    if (strncmp(argv[i], key, key_len) != 0)
        return 0;

    if (argv[i][key_len] == '=') {
        *value = argv[i] + key_len + 1;
        *consumed = 0;
        return 1;
    }

    if (argv[i][key_len] == '\0') {
        if (i + 1 >= argc)
            return -1;
        *value = argv[i + 1];
        *consumed = 1;
        return 1;
    }

    return 0;
}

static int parse_int_value(const char *text, int *out) {
    char *end = NULL;
    long value;

    if (!text || !*text || !out)
        return -1;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value < INT_MIN || value > INT_MAX)
        return -1;

    *out = (int)value;
    return 0;
}

int bench_parse_common_arg(BenchConfig *cfg, int argc, char **argv, int *i) {
    const char *v = NULL;
    int consumed = 0;
    int parsed = 0;

    if (strcmp(argv[*i], "--full") == 0) {
        cfg->repeats = 5;
        cfg->warmup = 2;
        return 1;
    }
    if (strcmp(argv[*i], "--no-header") == 0) {
        cfg->no_header = 1;
        return 1;
    }
    if (strcmp(argv[*i], "--append") == 0) {
        cfg->append = 1;
        return 1;
    }
    if (strcmp(argv[*i], "--overwrite") == 0) {
        cfg->append = 0;
        return 1;
    }

#define PARSE_INT(KEY, FIELD) \
    do { \
        parsed = arg_value(argc, argv, *i, KEY, &v, &consumed); \
        if (parsed < 0) return -1; \
        if (parsed) { \
            int value; \
            if (parse_int_value(v, &value) != 0) return -1; \
            cfg->FIELD = value; *i += consumed; return 1; \
        } \
    } while (0)

#define PARSE_STR(KEY, FIELD) \
    do { \
        parsed = arg_value(argc, argv, *i, KEY, &v, &consumed); \
        if (parsed < 0) return -1; \
        if (parsed) { cfg->FIELD = v; *i += consumed; return 1; } \
    } while (0)

    PARSE_STR("--csv", csv_path);
    PARSE_STR("--experiment", experiment);
    PARSE_STR("--variant", variant);
    PARSE_STR("--thread-list", thread_list);
    PARSE_INT("--samples", samples);
    PARSE_INT("--input", input);
    PARSE_INT("--hidden", hidden);
    PARSE_INT("--output", output);
    PARSE_INT("--epochs", epochs);
    PARSE_INT("--batch", batch);
    PARSE_INT("--repeats", repeats);
    PARSE_INT("--warmup", warmup);
    PARSE_INT("--tile", tile);
    PARSE_INT("--threads", max_threads);
    PARSE_INT("--report-loss-each-epoch", report_loss_each_epoch);

#undef PARSE_INT
#undef PARSE_STR

    return 0;
}

int bench_parse_int_list(const char *s, int *out, int max_count) {
    int count = 0;
    const char *p = s;

    if (!s || !*s || !out || max_count <= 0)
        return 0;

    while (*p) {
        char *end = NULL;
        if (count >= max_count)
            return 0;
        errno = 0;
        long value = strtol(p, &end, 10);
        if (errno == ERANGE || end == p || value <= 0 || value > INT_MAX)
            return 0;
        out[count++] = (int)value;
        if (*end == ',') {
            p = end + 1;
            if (*p == '\0')
                return 0;
        } else if (*end == '\0') {
            break;
        } else {
            return 0;
        }
    }

    return count;
}

static int validation_error(char *error, size_t error_size,
                            const char *message) {
    if (error && error_size > 0)
        snprintf(error, error_size, "%s", message);
    return -1;
}

int bench_validate_config(const BenchConfig *cfg, int require_threads,
                          int require_tile, char *error, size_t error_size) {
    if (!cfg)
        return validation_error(error, error_size, "null benchmark config");
    if (cfg->samples <= 0 || cfg->input <= 0 || cfg->hidden <= 0 ||
        cfg->output <= 0 || cfg->epochs <= 0 || cfg->batch <= 0)
        return validation_error(error, error_size,
                                "samples, dimensions, epochs, and batch must be positive");
    if (cfg->output > cfg->input)
        return validation_error(error, error_size,
                                "output size must not exceed input size");
    if (cfg->repeats <= 0 || cfg->warmup < 0)
        return validation_error(error, error_size,
                                "repeats must be positive and warmup non-negative");
    if (cfg->report_loss_each_epoch != 0 && cfg->report_loss_each_epoch != 1)
        return validation_error(error, error_size,
                                "report-loss-each-epoch must be 0 or 1");
    if (require_threads && cfg->max_threads <= 0)
        return validation_error(error, error_size,
                                "thread count must be positive");
    if (require_tile && cfg->tile <= 0)
        return validation_error(error, error_size,
                                "tile size must be positive");

    if ((size_t)cfg->input > (size_t)INT_MAX / (size_t)cfg->hidden ||
        (size_t)cfg->hidden > (size_t)INT_MAX / (size_t)cfg->output)
        return validation_error(error, error_size,
                                "network dimensions exceed supported integer indexing");
    if ((size_t)cfg->batch > (size_t)INT_MAX / (size_t)cfg->input ||
        (size_t)cfg->batch > (size_t)INT_MAX / (size_t)cfg->hidden ||
        (size_t)cfg->batch > (size_t)INT_MAX / (size_t)cfg->output)
        return validation_error(error, error_size,
                                "batch dimensions exceed supported integer indexing");

    if (error && error_size > 0)
        error[0] = '\0';
    return 0;
}

FILE *bench_open_csv(const BenchConfig *cfg) {
    const char *mode = cfg->append ? "a" : "w";
    if (!cfg->csv_path)
        return stdout;
    return fopen(cfg->csv_path, mode);
}

double bench_wtime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int bench_ceil_div(int a, int b) {
    if (a < 0 || b <= 0)
        return 0;
    return a / b + (a % b != 0);
}

void bench_write_csv_header(FILE *f) {
    fprintf(f, "experiment,backend,variant,samples,input_size,hidden_size,"
            "output_size,epochs,batch_size,num_threads,num_ranks,tile_size,"
            "repetition,warmup_reps,t_train,t_loss_eval,t_total,final_loss,"
            "updates_per_epoch,global_batch_size,last_global_batch_size\n");
}

void bench_write_csv_row(FILE *f, const BenchResult *row) {
    fprintf(f, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.9f,%.9f,%.9f,%.9f,%d,%d,%d\n",
            row->experiment ? row->experiment : "",
            row->backend ? row->backend : "",
            row->variant ? row->variant : "",
            row->samples,
            row->input_size,
            row->hidden_size,
            row->output_size,
            row->epochs,
            row->batch_size,
            row->num_threads,
            row->num_ranks,
            row->tile_size,
            row->repetition,
            row->warmup_reps,
            row->t_train,
            row->t_loss_eval,
            row->t_total,
            row->final_loss,
            row->updates_per_epoch,
            row->global_batch_size,
            row->last_global_batch_size);
}
