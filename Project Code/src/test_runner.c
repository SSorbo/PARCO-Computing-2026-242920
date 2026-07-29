#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "test_suites.h"

typedef struct {
    const char* name;
    int (*run)(void);
} TestSuite;

int main(int argc, char** argv) {
    /* Parse optional --suite filter */
    const char* filter = NULL;
    int verbose = 0;
    unsigned seed = 12345U;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--suite") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing --suite value\n");
                return 2;
            }
            filter = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing --seed value\n");
                return 2;
            }
            char *end = NULL;
            errno = 0;
            unsigned long parsed = strtoul(argv[++i], &end, 10);
            if (errno == ERANGE || end == argv[i] || *end != '\0' ||
                parsed > UINT_MAX) {
                fprintf(stderr, "Invalid --seed value\n");
                return 2;
            }
            seed = (unsigned)parsed;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else {
            fprintf(stderr, "Unknown test option: %s\n", argv[i]);
            return 2;
        }
    }

    /* Seed RNG deterministically so failures are reproducible. */
    srand(seed);

    printf("=== Test Suite Runner ===\n");
    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    if (verbose) {
        printf("RNG seed: %u\n", seed);
    }

    TestSuite suites[] = {
        {"Numerical Correctness", run_correctness_tests},
        {"OpenMP Determinism",   run_omp_kernel_tests},
        {"Gradient Correctness", run_nn_gradient_tests},
        {"Memory Safety",        run_memory_safety_tests},
        {"Training Reproducibility", run_reproducibility_tests},
    };
    int num_suites = sizeof(suites) / sizeof(TestSuite);

    int total_failed = 0;
    int suites_run = 0;

    for (int i = 0; i < num_suites; i++) {
        if (filter && strstr(suites[i].name, filter) == NULL)
            continue;

        printf("\n[%d/%d] %s\n", suites_run + 1,
               filter ? 1 : num_suites, suites[i].name);

        int failed = suites[i].run();
        if (failed > 0) {
            printf("\n>>> %s: %d FAILURES <<<\n", suites[i].name, failed);
        } else {
            printf("\n--- %s: ALL PASSED ---\n", suites[i].name);
        }
        total_failed += failed;
        suites_run++;
    }

    printf("\n========================================\n");
    if (total_failed == 0) {
        printf("  ALL TEST SUITES PASSED (%d suites)\n", suites_run);
    } else {
        printf("  %d FAILURES across %d suites\n", total_failed, suites_run);
    }
    printf("========================================\n");

    return total_failed > 0 ? 1 : 0;
}
