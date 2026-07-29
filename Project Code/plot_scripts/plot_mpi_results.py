#!/usr/bin/env python3
"""Plot single-node MPI strong scaling or tile sensitivity.

Strong-scaling plots are generated only for a fixed-tile rank sweep. A tile
sweep generates only the tile-sensitivity plot, preventing arbitrary tile
selection and stale, misleading output files.

Usage:
    python3 plot_scripts/plot_mpi_results.py [CSV] [--outdir DIR]
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
    config_columns,
    config_title,
    interval_yerr,
    iter_configs,
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


def is_tile_sweep(raw) -> bool:
    """Return true when at least one rank was tested at multiple tile sizes."""
    counts = raw.groupby("num_ranks")["tile_size"].nunique()
    return bool((counts >= 2).any())


def _paired_scaling(raw):
    base = raw[raw["num_ranks"] == 1][["repetition", "t_train"]].rename(
        columns={"t_train": "t_base"}
    )
    paired = raw.merge(base, on="repetition", how="inner")
    paired["speedup"] = paired["t_base"] / paired["t_train"]
    paired["efficiency"] = paired["speedup"] / paired["num_ranks"] * 100.0
    return paired


def plot_strong_scaling(raw, outdir: str) -> bool:
    """Plot fixed-tile MPI speedup and efficiency with repetition IQRs."""
    columns = config_columns(raw) + ["tile_size"]
    configs = [
        (config, subset)
        for config, subset in iter_configs(raw, columns)
        if subset["num_ranks"].nunique() >= 2
        and (subset["num_ranks"] == 1).any()
    ]
    if not configs:
        print("  (no fixed-tile MPI rank sweep with a 1-rank baseline)")
        remove_outputs(outdir, ["mpi_strong_scaling.png"])
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
        speed = aggregate_interval(paired, ["num_ranks"], "speedup")
        efficiency = aggregate_interval(paired, ["num_ranks"], "efficiency")
        ranks = speed["num_ranks"].to_numpy()

        ax_speed = axes[0, column]
        ax_eff = axes[1, column]
        ax_speed.errorbar(
            ranks,
            speed["speedup"],
            yerr=interval_yerr(speed, "speedup"),
            color=COLOR,
            marker="o",
            linewidth=2,
            capsize=3,
            label="MPI median ± IQR",
        )
        ax_speed.plot(
            [ranks.min(), ranks.max()],
            [ranks.min(), ranks.max()],
            color="gray",
            linestyle=":",
            label="ideal linear",
        )
        ax_speed.set_ylabel("Speedup vs. 1 MPI rank")
        ax_speed.set_title(
            f"{config_title(config, include_batch=True)}\ntile={int(config['tile_size'])}"
        )
        ax_speed.grid(True, alpha=0.3)
        ax_speed.legend(fontsize=8)

        ax_eff.errorbar(
            efficiency["num_ranks"],
            efficiency["efficiency"],
            yerr=interval_yerr(efficiency, "efficiency"),
            color="#d62728",
            marker="s",
            linewidth=2,
            capsize=3,
        )
        ax_eff.axhline(100.0, color="gray", linestyle=":")
        ax_eff.set_ylabel("Parallel efficiency (%)")
        ax_eff.set_ylim(bottom=0)
        ax_eff.grid(True, alpha=0.3)
        set_power_of_two_axis(ax_eff, ranks, "MPI ranks")

    fig.suptitle("Single-Node MPI Strong Scaling", fontsize=15)
    save_figure(fig, outdir, "mpi_strong_scaling.png")
    return True


def plot_throughput(raw, outdir: str) -> bool:
    """Plot one absolute throughput metric for each fixed MPI workload."""
    raw = raw.copy()
    raw["gflops"] = (
        raw["samples"]
        * raw["epochs"]
        * (
            4 * raw["input_size"] * raw["hidden_size"]
            + 6 * raw["hidden_size"] * raw["output_size"]
        )
        / (raw["t_train"] * 1e9)
    )
    columns = config_columns(raw) + ["tile_size"]
    configs = [
        (config, subset)
        for config, subset in iter_configs(raw, columns)
        if subset["num_ranks"].nunique() >= 2
    ]
    if not configs:
        print("  (no fixed-tile MPI rank sweep; skipping throughput)")
        remove_outputs(outdir, ["mpi_throughput.png"])
        return False

    fig, axes = plt.subplots(
        1, len(configs), figsize=(6.2 * len(configs), 4.8), squeeze=False
    )
    for index, (config, subset) in enumerate(configs):
        summary = aggregate_interval(subset, ["num_ranks"], "gflops")
        ax = axes[0, index]
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
        ax.set_title(f"{config_title(config)}\ntile={int(config['tile_size'])}")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Single-Node MPI Absolute Throughput", fontsize=15)
    save_figure(fig, outdir, "mpi_throughput.png")
    return True


def plot_tile_scaling(raw, outdir: str) -> bool:
    """Plot tile sensitivity in one facet per rank so scales stay readable."""
    rank_counts = raw.groupby("num_ranks")["tile_size"].nunique()
    valid_ranks = sorted(rank_counts[rank_counts >= 2].index)
    if not valid_ranks:
        print("  (fewer than two tile sizes per rank; skipping tile scaling)")
        remove_outputs(outdir, ["mpi_tile_scaling.png"])
        return False

    fig, axes = plt.subplots(
        1, len(valid_ranks), figsize=(5.2 * len(valid_ranks), 4.8), squeeze=False
    )
    for index, ranks in enumerate(valid_ranks):
        subset = raw[raw["num_ranks"] == ranks]
        summary = aggregate_interval(subset, ["tile_size"], "t_train")
        summary = summary.sort_values("tile_size")
        ax = axes[0, index]
        ax.errorbar(
            summary["tile_size"],
            summary["t_train"],
            yerr=interval_yerr(summary, "t_train"),
            color=COLOR,
            marker="o",
            linewidth=2,
            capsize=3,
        )
        best = summary.loc[summary["t_train"].idxmin()]
        ax.scatter(
            [best["tile_size"]],
            [best["t_train"]],
            s=90,
            color="#ff7f0e",
            edgecolor="black",
            zorder=3,
            label=f"best tested: {int(best['tile_size'])}",
        )
        set_power_of_two_axis(ax, summary["tile_size"], "Tile size")
        ax.set_ylabel("Training time (s)")
        ax.set_title(f"{int(ranks)} MPI rank{'s' if ranks != 1 else ''}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    first = raw.iloc[0]
    fig.suptitle(
        "MPI Tile-Size Sensitivity\n"
        f"N={int(first['samples'])}, "
        f"{int(first['input_size'])}\u2192{int(first['hidden_size'])}"
        f"\u2192{int(first['output_size'])}, single node",
        fontsize=15,
    )
    save_figure(fig, outdir, "mpi_tile_scaling.png")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="results/exp3_mpi_strong.csv")
    parser.add_argument("--outdir", default="plots/exp3_mpi_strong")
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
    if is_tile_sweep(raw):
        print("Detected a tile sweep; generating only tile sensitivity.")
        remove_outputs(args.outdir, ["mpi_strong_scaling.png", "mpi_throughput.png"])
        generated = [plot_tile_scaling(raw, args.outdir)]
    else:
        print("Detected a fixed-tile rank sweep; generating scaling and throughput.")
        remove_outputs(args.outdir, ["mpi_tile_scaling.png"])
        generated = [
            plot_strong_scaling(raw, args.outdir),
            plot_throughput(raw, args.outdir),
        ]
    print(f"Done. {sum(generated)} applicable plot(s) generated in {args.outdir}/")


if __name__ == "__main__":
    main()
