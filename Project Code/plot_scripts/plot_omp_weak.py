#!/usr/bin/env python3
"""Generate report-ready OpenMP weak-scaling figures.

The main figure combines wall time and baseline-normalized efficiency. A single
supporting throughput plot reports estimated GEMM GFLOP/s; samples/s is omitted
because work per sample changes with hidden width in this experiment.

Usage:
    python3 plot_scripts/plot_omp_weak.py [CSV] [--outdir DIR]
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
    "num_threads",
    "repetition",
    "t_train",
]


def prepare(raw):
    data = raw[raw["variant"] == "omp"].copy()
    if data.empty:
        raise ValueError("CSV contains no OpenMP rows")
    data["work"] = (
        data["samples"]
        * data["epochs"]
        * (
            4 * data["input_size"] * data["hidden_size"]
            + 6 * data["hidden_size"] * data["output_size"]
        )
    )
    data["gflops"] = data["work"] / (data["t_train"] * 1e9)

    minimum = data["num_threads"].min()
    base = data[data["num_threads"] == minimum][
        ["repetition", "num_threads", "work", "t_train"]
    ].rename(
        columns={
            "num_threads": "base_threads",
            "work": "base_work",
            "t_train": "base_time",
        }
    )
    data = data.merge(base, on="repetition", how="inner")
    worker_ratio = data["num_threads"] / data["base_threads"]
    base_rate = data["base_work"] / data["base_time"]
    actual_rate = data["work"] / data["t_train"]
    data["weak_efficiency"] = actual_rate / (worker_ratio * base_rate) * 100.0
    return data


def plot_scaling(data, outdir: str) -> None:
    time = aggregate_interval(
        data, ["num_threads", "hidden_size"], "t_train"
    ).sort_values("num_threads")
    efficiency = aggregate_interval(
        data, ["num_threads", "hidden_size"], "weak_efficiency"
    ).sort_values("num_threads")

    fig, (ax_time, ax_eff) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_time.errorbar(
        time["num_threads"],
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
            f"h={int(row['hidden_size'])}",
            (row["num_threads"], row["t_train"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax_time.set_ylabel("Training time (s)")
    ax_time.set_title("Wall time")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(fontsize=8)

    ax_eff.errorbar(
        efficiency["num_threads"],
        efficiency["weak_efficiency"],
        yerr=interval_yerr(efficiency, "weak_efficiency"),
        color="#9467bd",
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

    set_power_of_two_axis(ax_time, time["num_threads"], "OpenMP threads")
    set_power_of_two_axis(ax_eff, time["num_threads"], "OpenMP threads")
    first = data.iloc[0]
    fig.suptitle(
        "OpenMP Weak Scaling: Model Width Grows with Threads\n"
        f"N={int(first['samples'])}, input={int(first['input_size'])}, "
        f"{int(first['epochs'])} epochs",
        fontsize=15,
    )
    save_figure(fig, outdir, "omp_weak_scaling.png")


def plot_throughput(data, outdir: str) -> None:
    summary = aggregate_interval(data, ["num_threads"], "gflops").sort_values(
        "num_threads"
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        summary["num_threads"],
        summary["gflops"],
        yerr=interval_yerr(summary, "gflops"),
        color=COLOR,
        marker="o",
        linewidth=2,
        capsize=3,
    )
    set_power_of_two_axis(ax, summary["num_threads"], "OpenMP threads")
    ax.set_ylabel("Estimated GEMM GFLOP/s")
    ax.set_title("OpenMP Weak Scaling: Absolute Throughput")
    ax.grid(True, alpha=0.3)
    save_figure(fig, outdir, "omp_weak_throughput.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="results/exp4_omp_weak.csv")
    parser.add_argument("--outdir", default="plots/exp4_omp_weak")
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
    remove_outputs(args.outdir, ["omp_weak_time.png", "omp_weak_efficiency.png"])
    print(f"Loading {args.csv}: {len(raw)} rows")
    plot_scaling(data, args.outdir)
    plot_throughput(data, args.outdir)
    print(f"Done. 2 plots generated in {args.outdir}/")


if __name__ == "__main__":
    main()
