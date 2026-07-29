#ifndef TEST_SUITES_H
#define TEST_SUITES_H

/* Each suite returns its number of failed checks. */
int run_correctness_tests(void);
int run_omp_kernel_tests(void);
int run_nn_gradient_tests(void);
int run_memory_safety_tests(void);
int run_reproducibility_tests(void);

#endif
