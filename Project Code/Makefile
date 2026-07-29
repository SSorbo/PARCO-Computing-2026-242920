CC = gcc
MPICC = mpicc
CFLAGS ?= -O3 -std=c11 -D_POSIX_C_SOURCE=200112L -Iinclude -Wall -Wextra -Wpedantic
OMPFLAGS ?= -fopenmp
LDFLAGS ?= -lm
SEQ_BIN ?= seq_train
OMP_BIN ?= omp_train
MPI_BIN ?= mpi_nn_train
TEST_BIN ?= test_runner
MPI_TEST_BIN ?= test_mpi_correctness
MPIRUN ?= mpirun
PYTHON ?= python3

SEQ_SOURCES = src/main_seq_train.c src/nn.c src/math_utils.c src/train_seq.c src/train_utils.c src/bench_cli.c
OMP_SOURCES = src/main_omp_train.c src/nn.c src/math_utils.c src/math_omp.c src/train_omp.c src/train_utils.c src/bench_cli.c
MPI_SOURCES = src/main_mpi_train.c src/train_mpi.c src/nn.c src/math_utils.c src/train_utils.c src/bench_cli.c
TEST_SOURCES = src/test_runner.c src/test_correctness.c src/test_omp_determinism.c src/test_nn_gradients.c src/test_memory_safety.c src/test_reproducibility.c src/nn.c src/math_utils.c src/math_omp.c src/train_seq.c src/train_omp.c src/train_utils.c
MPI_TEST_SOURCES = src/test_mpi_correctness.c src/train_mpi.c src/train_seq.c src/nn.c src/math_utils.c src/train_utils.c

PLOT_DIRS = \
	plots/exp1_seq_baseline \
	plots/exp2_omp_strong \
	plots/exp3_mpi_strong \
	plots/exp4_omp_weak \
	plots/exp5_mpi_weak \
	plots/exp6_mpi_tile \
	plots/exp7_batch \
	plots/cross_experiment

.PHONY: all seq omp mpi test test-mpi dirs plots clean

all: seq omp mpi

seq: $(SEQ_SOURCES)
	$(CC) $(CFLAGS) $(SEQ_SOURCES) -o $(SEQ_BIN) $(LDFLAGS)

omp: $(OMP_SOURCES)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(OMP_SOURCES) -o $(OMP_BIN) $(LDFLAGS)

mpi: $(MPI_SOURCES)
	$(MPICC) $(CFLAGS) $(MPI_SOURCES) -o $(MPI_BIN) $(LDFLAGS)

test: $(TEST_SOURCES)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(TEST_SOURCES) -o $(TEST_BIN) $(LDFLAGS)
	./$(TEST_BIN)

test-mpi: $(MPI_TEST_SOURCES)
	$(MPICC) $(CFLAGS) $(MPI_TEST_SOURCES) -o $(MPI_TEST_BIN) $(LDFLAGS)
	$(MPIRUN) -np 1 ./$(MPI_TEST_BIN)
	$(MPIRUN) -np 2 ./$(MPI_TEST_BIN)

dirs:
	mkdir -p artifacts results $(PLOT_DIRS)

plots: dirs
	$(PYTHON) plot_scripts/plot_omp_results.py results/exp1_seq_baseline.csv --outdir plots/exp1_seq_baseline
	$(PYTHON) plot_scripts/plot_omp_results.py results/exp2_omp_strong.csv --outdir plots/exp2_omp_strong
	$(PYTHON) plot_scripts/plot_mpi_results.py results/exp3_mpi_strong.csv --outdir plots/exp3_mpi_strong
	$(PYTHON) plot_scripts/plot_omp_weak.py results/exp4_omp_weak.csv --outdir plots/exp4_omp_weak
	$(PYTHON) plot_scripts/plot_mpi_weak.py results/exp5_mpi_weak.csv --outdir plots/exp5_mpi_weak
	$(PYTHON) plot_scripts/plot_mpi_results.py results/exp6_mpi_tile.csv --outdir plots/exp6_mpi_tile
	$(PYTHON) plot_scripts/plot_cross_experiment.py --results-dir results --outdir plots/cross_experiment
	$(PYTHON) plot_scripts/plot_exp7_batch.py --results-dir results --outdir plots/exp7_batch

clean:
	rm -f seq_train omp_train mpi_nn_train test_runner test_mpi_correctness seq_train_* omp_train_* mpi_nn_train_*
	rm -rf __pycache__ plot_scripts/__pycache__ *.dSYM
