#!/usr/bin/env python3
"""Generate report-ready single-node MPI weak-scaling figures.

The main figure combines wall time and baseline-normalized efficiency. A
supporting plot reports one non-redundant absolute throughput metric.

Usage:
    python3 plot_scripts/plot_mpi_weak.py [CSV] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_common import (
    aggregate_interval,
    apply_style,
    interval_yerr,
    load_csv,
    remove_outputs,
    save_figure,
    set_power_of_two_axis,
)


COLOR = "#1f77b4"
REQUIRED = [
    "experiment",
    "variant",
    "samples",
    "input_size",
    "hidden_size",
    "output_size",
    "epochs",
    "batch_size",
    "tile_size",
    "num_ranks",
    "repetition",
    "t_train",
]


def prepare(raw):
    data = raw.copy()
    data["samples_per_rank"] = data["samples"] / data["num_ranks"]
    if data["samples_per_rank"].nunique() != 1:
        raise ValueError("MPI weak-scaling data do not keep samples/rank fixed")

    data["work"] = (
        data["samples"]
        * data["epochs"]
        * (
            4 * data["input_size"] * data["hidden_size"]
            + 6 * data["hidden_size"] * data["output_size"]
        )
    )
    data["gflops"] = data["work"] / (data["t_train"] * 1e9)

    minimum = data["num_ranks"].min()
    base = data[data["num_ranks"] == minimum][
        ["repetition", "num_ranks", "work", "t_train"]
    ].rename(
        columns={
            "num_ranks": "base_ranks",
            "work": "base_work",
            "t_train": "base_time",
        }
    )
    data = data.merge(base, on="repetition", how="inner")
    worker_ratio = data["num_ranks"] / data["base_ranks"]
    base_rate = data["base_work"] / data["base_time"]
    actual_rate = data["work"] / data["t_train"]
    data["weak_efficiency"] = actual_rate / (worker_ratio * base_rate) * 100.0
    return data


def plot_scaling(data, outdir: str) -> None:
    time = aggregate_interval(
        data, ["num_ranks", "samples", "samples_per_rank"], "t_train"
    ).sort_values("num_ranks")
    efficiency = aggregate_interval(
        data, ["num_ranks"], "weak_efficiency"
    ).sort_values("num_ranks")

    fig, (ax_time, ax_eff) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_time.errorbar(
        time["num_ranks"],
        time["t_train"],
        yerr=interval_yerr(time, "t_train"),
        color=COLOR,
        marker="o",
        linewidth=2,
        capsize=3,
    )
    baseline = time.iloc[0]
    ax_time.axhline(
        baseline["t_train"],
        color="gray",
        linestyle=":",
        label="ideal flat time",
    )
    for _, row in time.iterrows():
        ax_time.annotate(
            f"N={int(row['samples'])}",
            (row["num_ranks"], row["t_train"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax_time.set_ylabel("Training time (s)")
    ax_time.set_title("Wall time")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=8)

    ax_eff.errorbar(
        efficiency["num_ranks"],
        efficiency["weak_efficiency"],
        yerr=interval_yerr(efficiency, "weak_efficiency"),
        color="#d62728",
        marker="s",
        linewidth=2,
        capsize=3,
    )
    ax_eff.axhline(100.0, color="gray", linestyle=":", label="ideal (100%)")
    ax_eff.set_ylabel("Weak-scaling efficiency (%)")
    ax_eff.set_ylim(bottom=0)
    ax_eff.set_title("Baseline-normalized efficiency")
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend(fontsize=8)

    set_power_of_two_axis(ax_time, time["num_ranks"], "MPI ranks")
    set_power_of_two_axis(ax_eff, time["num_ranks"], "MPI ranks")
    first = data.iloc[0]
    fig.suptitle(
        "Single-Node MPI Weak Scaling: Dataset Grows with Ranks\n"
        f"{int(first['samples_per_rank'])} samples/rank, "
        f"{int(first['input_size'])}\u2192{int(first['hidden_size'])}"
        f"\u2192{int(first['output_size'])}, tile={int(first['tile_size'])}",
        fontsize=15,
    )
    save_figure(fig, outdir, "mpi_weak_scaling.png")


def plot_throughput(data, outdir: str) -> None:
    summary = aggregate_interval(data, ["num_ranks"], "gflops").sort_values(
        "num_ranks"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        summary["num_ranks"],
        summary["gflops"],
        yerr=interval_yerr(summary, "gflops"),
        color=COLOR,
        marker="o",
        linewidth=2,
        capsize=3,
    )
    set_power_of_two_axis(ax, summary["num_ranks"], "MPI ranks")
    ax.set_ylabel("Estimated GEMM GFLOP/s")
    ax.set_title("Single-Node MPI Weak Scaling: Absolute Throughput")
    ax.grid(True, alpha=0.3)
    save_figure(fig, outdir, "mpi_weak_throughput.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="results/exp5_mpi_weak.csv")
    parser.add_argument("--outdir", default="plots/exp5_mpi_weak")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found.", file=sys.stderr)
        raise SystemExit(1)
    apply_style()
    try:
        raw = load_csv(args.csv, REQUIRED)
        data = prepare(raw)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    os.makedirs(args.outdir, exist_ok=True)
    remove_outputs(args.outdir, ["mpi_weak_time.png", "mpi_weak_efficiency.png"])
    print(f"Loading {args.csv}: {len(raw)} rows")
    plot_scaling(data, args.outdir)
    plot_throughput(data, args.outdir)
    print(f"Done. 2 plots generated in {args.outdir}/")


if __name__ == "__main__":
    main()
