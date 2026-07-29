# Parallel Neural-Network Benchmark

This directory contains the supported implementation of a two-layer sigmoid
network benchmark with naive sequential, tiled sequential, OpenMP, and
single-node MPI training backends.

## Build and correctness checks

```bash
make seq
make omp
make mpi
make test
make test-mpi
```

`make test-mpi` requires `mpicc` and `mpirun`. It checks one-rank equivalence
with tiled sequential training and exact replica synchronization after unequal
local final mini-batches at two ranks.

On macOS with Homebrew LLVM/OpenMP, use:

```bash
make test CC=/opt/homebrew/opt/llvm/bin/clang
```

## Supported executables

- `seq_train`: `--variant=naive`, `--variant=seq`, or `--variant=both`.
- `omp_train`: OpenMP backend with `--thread-list=1,2,4,...` or
  `--threads=N`.
- `mpi_nn_train`: synchronous replicated-model MPI backend with runtime
  `--tile=N`.

## Cluster experiments

Run all commands from the `Project Code/` directory. Before submitting a job,
run `make dirs`; this creates `artifacts/`, `results/`, and every expected plot
directory. The `artifacts/` directory must exist before `qsub` opens the PBS
output files. Submit jobs as `qsub pbs_scripts/<job>.pbs`; their standard
output and error files are written under `artifacts/`.

| Experiment | PBS job | Output CSV |
|---|---|---|
| Sequential baselines | `pbs_scripts/submit_exp1_seq_baseline.pbs` | `results/exp1_seq_baseline.csv` |
| OpenMP strong scaling | `pbs_scripts/submit_exp2_omp_strong.pbs` | `results/exp2_omp_strong.csv` |
| Single-node MPI strong scaling | `pbs_scripts/submit_exp3_mpi_strong.pbs` | `results/exp3_mpi_strong.csv` |
| OpenMP weak scaling | `pbs_scripts/submit_exp4_omp_weak.pbs` | `results/exp4_omp_weak.csv` |
| Single-node MPI weak scaling | `pbs_scripts/submit_exp5_mpi_weak.pbs` | `results/exp5_mpi_weak.csv` |
| MPI tile sensitivity | `pbs_scripts/submit_exp6_mpi_tile.pbs` | `results/exp6_mpi_tile.csv` |
| Sequential/OpenMP batch sensitivity | `pbs_scripts/submit_exp7_batch_seq_omp.pbs` | `results/exp7_batch_seq_omp.csv` |
| MPI batch sensitivity | `pbs_scripts/submit_exp7_batch_mpi.pbs` | `results/exp7_batch_mpi.csv` |

All supplied MPI jobs request one node and use shared-memory/self transports;
their results must not be described as multi-node scaling.

## Plotting

Install direct Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Regenerate every plot in the standard directory structure:

```bash
make plots
```

Generate the experiment-local plots:

```bash
python3 plot_scripts/plot_omp_results.py results/exp1_seq_baseline.csv \
  --outdir plots/exp1_seq_baseline
python3 plot_scripts/plot_omp_results.py results/exp2_omp_strong.csv \
  --outdir plots/exp2_omp_strong
python3 plot_scripts/plot_mpi_results.py results/exp3_mpi_strong.csv \
  --outdir plots/exp3_mpi_strong
python3 plot_scripts/plot_omp_weak.py results/exp4_omp_weak.csv \
  --outdir plots/exp4_omp_weak
python3 plot_scripts/plot_mpi_weak.py results/exp5_mpi_weak.csv \
  --outdir plots/exp5_mpi_weak
python3 plot_scripts/plot_mpi_results.py results/exp6_mpi_tile.csv \
  --outdir plots/exp6_mpi_tile
```

Generate the cross-experiment and batch-sensitivity figures:

```bash
python3 plot_scripts/plot_cross_experiment.py \
  --results-dir results --outdir plots/cross_experiment
python3 plot_scripts/plot_exp7_batch.py \
  --results-dir results --outdir plots/exp7_batch
```
