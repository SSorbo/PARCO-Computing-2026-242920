#!/usr/bin/env python3
"""Plot sequential baselines or OpenMP strong-scaling results.

The script is experiment-aware:

- variant comparisons require at least two variants for the same workload;
- strong scaling and throughput require OpenMP data at two or more threads;
- every workload is shown in a facet instead of silently selecting one;
- medians are accompanied by interquartile ranges from timed repetitions.

Usage:
    python3 plot_scripts/plot_omp_results.py [CSV] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    aggregate_interval,
    apply_style,
    config_columns,
    config_title,
    interval_yerr,
    iter_configs,
    load_csv,
    remove_outputs,
    save_figure,
    set_power_of_two_axis,
)


VARIANT_COLORS = {"naive": "#d62728", "seq": "#2ca02c", "omp": "#1f77b4"}
VARIANT_MARKERS = {"naive": "s", "seq": "D", "omp": "o"}
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


def _paired_scaling(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute repetition-paired speedup and efficiency from one-thread runs."""
    base = raw[raw["num_threads"] == 1][["repetition", "t_train"]].rename(
        columns={"t_train": "t_base"}
    )
    paired = raw.merge(base, on="repetition", how="inner")
    paired["speedup"] = paired["t_base"] / paired["t_train"]
    paired["efficiency"] = paired["speedup"] / paired["num_threads"] * 100.0
    return paired


def plot_strong_scaling(raw: pd.DataFrame, outdir: str) -> bool:
    """Plot speedup and efficiency for every supported OpenMP workload."""
    omp = raw[raw["variant"] == "omp"].copy()
    columns = config_columns(omp)
    configs = [
        (config, subset)
        for config, subset in iter_configs(omp, columns)
        if subset["num_threads"].nunique() >= 2
        and (subset["num_threads"] == 1).any()
    ]
    if not configs:
        print("  (no OpenMP workload with a 1-thread baseline and >=2 thread counts)")
        remove_outputs(outdir, ["strong_scaling.png"])
        return False

    fig, axes = plt.subplots(
        2,
        len(configs),
        figsize=(6.3 * len(configs), 8),
        squeeze=False,
        sharex="col",
    )
    for column, (config, subset) in enumerate(configs):
        paired = _paired_scaling(subset)
        speed = aggregate_interval(paired, ["num_threads"], "speedup")
        efficiency = aggregate_interval(paired, ["num_threads"], "efficiency")

        ax_speed = axes[0, column]
        ax_eff = axes[1, column]
        threads = speed["num_threads"].to_numpy()
        ax_speed.errorbar(
            threads,
            speed["speedup"],
            yerr=interval_yerr(speed, "speedup"),
            color=VARIANT_COLORS["omp"],
            marker=VARIANT_MARKERS["omp"],
            linewidth=2,
            capsize=3,
            label="OpenMP median ± IQR",
        )
        ax_speed.plot(
            [threads.min(), threads.max()],
            [threads.min(), threads.max()],
            color="gray",
            linestyle=":",
            label="ideal linear",
        )
        ax_speed.set_ylabel("Speedup vs. 1 OpenMP thread")
        ax_speed.set_title(config_title(config, include_batch=True))
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend(fontsize=8)

        ax_eff.errorbar(
            efficiency["num_threads"],
            efficiency["efficiency"],
            yerr=interval_yerr(efficiency, "efficiency"),
            color="#9467bd",
            marker="s",
            linewidth=2,
            capsize=3,
        )
        ax_eff.axhline(100.0, color="gray", linestyle=":", label="ideal (100%)")
        ax_eff.set_ylabel("Parallel efficiency (%)")
        ax_eff.set_ylim(bottom=0)
        ax_eff.grid(True, alpha=0.3)
        set_power_of_two_axis(ax_eff, threads, "OpenMP threads")

        best = subset.groupby("num_threads")["t_train"].median().idxmin()
        best_row = speed[speed["num_threads"] == best].iloc[0]
        ax_speed.annotate(
            f"best time: {int(best)} threads\n{best_row['speedup']:.1f}× speedup",
            (best, best_row["speedup"]),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            fontsize=8,
        )

    fig.suptitle("OpenMP Strong Scaling", fontsize=15)
    save_figure(fig, outdir, "strong_scaling.png")
    return True


def plot_variant_comparison(raw: pd.DataFrame, outdir: str) -> bool:
    """Compare two or more implementations for each fixed workload."""
    columns = config_columns(raw)
    configs = [
        (config, subset)
        for config, subset in iter_configs(raw, columns)
        if subset["variant"].nunique() >= 2
    ]
    if not configs:
        print("  (fewer than two variants per workload; skipping comparison)")
        remove_outputs(outdir, ["variant_comparison.png"])
        return False

    fig, axes = plt.subplots(
        1, len(configs), figsize=(6.2 * len(configs), 5.3), squeeze=False
    )
    order = ["naive", "seq", "omp"]
    for index, (config, subset) in enumerate(configs):
        ax = axes[0, index]
        summary = aggregate_interval(subset, ["variant"], "t_train")
        summary["sort"] = summary["variant"].map(
            {variant: position for position, variant in enumerate(order)}
        )
        summary = summary.sort_values(["sort", "variant"])
        x = np.arange(len(summary))
        colors = [VARIANT_COLORS.get(value, "#7f7f7f") for value in summary["variant"]]
        bars = ax.bar(
            x,
            summary["t_train"],
            yerr=interval_yerr(summary, "t_train"),
            capsize=4,
            color=colors,
            edgecolor="white",
        )
        ax.set_xticks(x, summary["variant"])
        ax.set_ylabel("Training time (s)")
        ax.set_title(config_title(config, include_batch=True))
        ax.grid(True, alpha=0.3, axis="y")

        baseline = summary[summary["variant"] == "naive"]
        baseline_time = baseline["t_train"].iloc[0] if not baseline.empty else None
        for bar, (_, row) in zip(bars, summary.iterrows()):
            label = f"{row['t_train']:.2f} s"
            if baseline_time and row["variant"] != "naive":
                label += f"\n{baseline_time / row['t_train']:.2f}×"
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

    fig.suptitle("Naive versus Tiled Sequential Training", fontsize=15)
    save_figure(fig, outdir, "variant_comparison.png")
    return True


def plot_throughput(raw: pd.DataFrame, outdir: str) -> bool:
    """Plot one non-redundant absolute throughput metric for OpenMP sweeps."""
    omp = raw[raw["variant"] == "omp"].copy()
    columns = config_columns(omp)
    configs = [
        (config, subset)
        for config, subset in iter_configs(omp, columns)
        if subset["num_threads"].nunique() >= 2
    ]
    if not configs:
        print("  (no OpenMP thread sweep; skipping throughput)")
        remove_outputs(outdir, ["throughput.png"])
        return False

    omp["gflops"] = (
        omp["samples"]
        * omp["epochs"]
        * (
            4 * omp["input_size"] * omp["hidden_size"]
            + 6 * omp["hidden_size"] * omp["output_size"]
        )
        / (omp["t_train"] * 1e9)
    )

    fig, axes = plt.subplots(
        1, len(configs), figsize=(6.2 * len(configs), 4.8), squeeze=False
    )
    for index, (config, _) in enumerate(configs):
        mask = pd.Series(True, index=omp.index)
        for column, value in config.items():
            mask &= omp[column] == value
        summary = aggregate_interval(omp[mask], ["num_threads"], "gflops")
        ax = axes[0, index]
        ax.errorbar(
            summary["num_threads"],
            summary["gflops"],
            yerr=interval_yerr(summary, "gflops"),
            color=VARIANT_COLORS["omp"],
            marker="o",
            linewidth=2,
            capsize=3,
        )
        set_power_of_two_axis(ax, summary["num_threads"], "OpenMP threads")
        ax.set_ylabel("Estimated GEMM GFLOP/s")
        ax.set_title(config_title(config))
        ax.grid(True, alpha=0.3)

    fig.suptitle("OpenMP Absolute Throughput", fontsize=15)
    save_figure(fig, outdir, "throughput.png")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="results/exp2_omp_strong.csv")
    parser.add_argument("--outdir", default="plots/exp2_omp_strong")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found.", file=sys.stderr)
        raise SystemExit(1)

    apply_style()
    try:
        raw = load_csv(args.csv, REQUIRED)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Loading {args.csv}: {len(raw)} rows")
    generated = [
        plot_strong_scaling(raw, args.outdir),
        plot_variant_comparison(raw, args.outdir),
        plot_throughput(raw, args.outdir),
    ]
    print(f"Done. {sum(generated)} applicable plot(s) generated in {args.outdir}/")


if __name__ == "__main__":
    main()
