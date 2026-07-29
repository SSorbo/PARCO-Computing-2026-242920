#!/usr/bin/env python3
"""Generate Experiment 7 batch-sensitivity figures.

Outputs:
    seq_omp_batch_runtime.png
    mpi_batch_runtime.png
    batch_semantics.png

The MPI runtime plot uses actual global batch size. The companion semantics
figure makes the different sequential/OpenMP and MPI interpretations of the CLI
batch value explicit.

Usage:
    python3 plot_scripts/plot_exp7_batch.py
        [--results-dir results] [--outdir plots/exp7_batch]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_common import (
    aggregate_interval,
    apply_style,
    interval_yerr,
    load_csv,
    save_figure,
    set_power_of_two_axis,
)


COLORS = {
    "seq": "#2ca02c",
    8: "#1f77b4",
    32: "#9467bd",
    64: "#ff7f0e",
}
REQUIRED = [
    "experiment",
    "backend",
    "variant",
    "samples",
    "input_size",
    "hidden_size",
    "output_size",
    "epochs",
    "batch_size",
    "num_threads",
    "num_ranks",
    "repetition",
    "t_train",
    "final_loss",
    "global_batch_size",
    "updates_per_epoch",
]


def plot_seq_omp_runtime(raw, outdir: str) -> None:
    summary = aggregate_interval(
        raw, ["variant", "num_threads", "batch_size"], "t_train"
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.2))

    seq = summary[summary["variant"] == "seq"].sort_values("batch_size")
    ax.errorbar(
        seq["batch_size"],
        seq["t_train"],
        yerr=interval_yerr(seq, "t_train"),
        color=COLORS["seq"],
        marker="D",
        linewidth=2,
        capsize=3,
        label="tiled sequential",
    )
    for threads in sorted(raw[raw["variant"] == "omp"]["num_threads"].unique()):
        data = summary[
            (summary["variant"] == "omp") & (summary["num_threads"] == threads)
        ].sort_values("batch_size")
        ax.errorbar(
            data["batch_size"],
            data["t_train"],
            yerr=interval_yerr(data, "t_train"),
            color=COLORS.get(int(threads), "#7f7f7f"),
            marker="o",
            linewidth=2,
            capsize=3,
            label=f"OpenMP, {int(threads)} threads",
        )

    set_power_of_two_axis(ax, summary["batch_size"], "Global batch size")
    ax.set_yscale("log")
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_title("Experiment 7: Sequential and OpenMP Batch Sensitivity")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, outdir, "seq_omp_batch_runtime.png")


def plot_mpi_runtime(raw, outdir: str) -> None:
    summary = aggregate_interval(
        raw, ["num_ranks", "global_batch_size"], "t_train"
    )
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for ranks in sorted(summary["num_ranks"].unique()):
        data = summary[summary["num_ranks"] == ranks].sort_values(
            "global_batch_size"
        )
        ax.errorbar(
            data["global_batch_size"],
            data["t_train"],
            yerr=interval_yerr(data, "t_train"),
            color=COLORS.get(int(ranks), "#7f7f7f"),
            marker="o",
            linewidth=2,
            capsize=3,
            label=f"{int(ranks)} ranks",
        )

    set_power_of_two_axis(
        ax, summary["global_batch_size"], "Actual global batch size"
    )
    ax.set_xlabel("Actual global batch size", labelpad=12)
    ax.tick_params(axis="x", labelsize=9, pad=5)
    ax.set_yscale("log")
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_title("Experiment 7: Single-Node MPI Batch Sensitivity")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save_figure(fig, outdir, "mpi_batch_runtime.png")


def _set_log2_y(ax, values, label: str) -> None:
    ticks = sorted({int(value) for value in values})
    ax.set_yscale("log", base=2)
    ax.set_yticks(ticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylabel(label)


def plot_semantics(shared_raw, mpi_raw, outdir: str) -> None:
    shared = (
        shared_raw[shared_raw["variant"] == "seq"]
        .drop_duplicates(["batch_size"])
        .sort_values("batch_size")
    )
    mpi = (
        mpi_raw.drop_duplicates(["batch_size", "num_ranks"])
        .sort_values(["num_ranks", "batch_size"])
    )

    fig, (ax_batch, ax_updates) = plt.subplots(1, 2, figsize=(12, 5))
    ax_batch.plot(
        shared["batch_size"],
        shared["global_batch_size"],
        color=COLORS["seq"],
        marker="D",
        linewidth=2,
        label="sequential/OpenMP",
    )
    ax_updates.plot(
        shared["batch_size"],
        shared["updates_per_epoch"],
        color=COLORS["seq"],
        marker="D",
        linewidth=2,
        label="sequential/OpenMP",
    )
    for ranks in sorted(mpi["num_ranks"].unique()):
        data = mpi[mpi["num_ranks"] == ranks]
        color = COLORS.get(int(ranks), "#7f7f7f")
        label = f"MPI, {int(ranks)} ranks"
        ax_batch.plot(
            data["batch_size"],
            data["global_batch_size"],
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )
        ax_updates.plot(
            data["batch_size"],
            data["updates_per_epoch"],
            color=color,
            marker="o",
            linewidth=2,
            label=label,
        )

    all_batches = sorted(shared["batch_size"].unique())
    set_power_of_two_axis(ax_batch, all_batches, "CLI batch value")
    set_power_of_two_axis(ax_updates, all_batches, "CLI batch value")
    _set_log2_y(
        ax_batch,
        list(shared["global_batch_size"]) + list(mpi["global_batch_size"]),
        "Actual global batch size",
    )
    _set_log2_y(
        ax_updates,
        list(shared["updates_per_epoch"]) + list(mpi["updates_per_epoch"]),
        "Updates per epoch",
    )
    ax_batch.set_title("Batch interpretation")
    ax_updates.set_title("Optimization work per epoch")
    for ax in (ax_batch, ax_updates):
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(
        "Experiment 7 Batch Semantics: MPI Uses a Local Batch",
        fontsize=15,
    )
    save_figure(fig, outdir, "batch_semantics.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--outdir", default="plots/exp7_batch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = Path(args.results_dir)
    shared_path = results / "exp7_batch_seq_omp.csv"
    mpi_path = results / "exp7_batch_mpi.csv"
    missing = [str(path) for path in (shared_path, mpi_path) if not path.exists()]
    if missing:
        print(f"Error: missing result files: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)

    apply_style()
    try:
        shared = load_csv(str(shared_path), REQUIRED)
        mpi = load_csv(str(mpi_path), REQUIRED)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    os.makedirs(args.outdir, exist_ok=True)
    plot_seq_omp_runtime(shared, args.outdir)
    plot_mpi_runtime(mpi, args.outdir)
    plot_semantics(shared, mpi, args.outdir)

    loss_min = min(shared["final_loss"].min(), mpi["final_loss"].min())
    loss_max = max(shared["final_loss"].max(), mpi["final_loss"].max())
    print(
        f"Final loss range is {loss_min:.8f}..{loss_max:.8f}; "
        "no uninformative loss plot was generated."
    )
    print(f"Done. 3 Experiment 7 plots generated in {args.outdir}/")


if __name__ == "__main__":
    main()
