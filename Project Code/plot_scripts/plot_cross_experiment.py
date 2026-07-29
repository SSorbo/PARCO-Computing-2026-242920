#!/usr/bin/env python3
"""Generate the highest-value comparisons across P2D experiments.

Outputs:
    seq_vs_omp.png
    omp_vs_mpi_strong.png
    mpi_tile_tuning_impact.png

Usage:
    python3 plot_scripts/plot_cross_experiment.py
        [--results-dir results] [--outdir plots/cross_experiment]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_common import (
    aggregate_interval,
    apply_style,
    interval_yerr,
    load_csv,
    save_figure,
    set_power_of_two_axis,
)


COLORS = {
    "naive": "#d62728",
    "seq": "#2ca02c",
    "omp": "#1f77b4",
    "mpi": "#ff7f0e",
}
WORKLOAD = [
    "samples",
    "input_size",
    "hidden_size",
    "output_size",
    "epochs",
    "batch_size",
]
BASE_REQUIRED = [
    "experiment",
    "variant",
    *WORKLOAD,
    "repetition",
    "t_train",
]


def workload_key(row) -> tuple:
    return tuple(int(row[column]) for column in WORKLOAD)


def workload_title(key: tuple) -> str:
    samples, input_size, hidden_size, output_size, epochs, batch = key
    name = "Canonical" if input_size == 1024 else "Large input"
    return (
        f"{name}\nN={samples}, {input_size}\u2192{hidden_size}\u2192{output_size}, "
        f"{epochs} epochs, batch={batch}"
    )


def matching_workloads(left, right) -> list[tuple]:
    left_keys = {workload_key(row) for _, row in left.iterrows()}
    right_keys = {workload_key(row) for _, row in right.iterrows()}
    return sorted(
        left_keys & right_keys,
        key=lambda key: (key[1] != 1024, key[1], key[0]),
    )


def select_workload(data, key: tuple):
    mask = pd.Series(True, index=data.index)
    for column, value in zip(WORKLOAD, key):
        mask &= data[column] == value
    return data[mask].copy()


def plot_seq_vs_omp(seq_raw, omp_raw, outdir: str) -> None:
    """Compare naive/tiled sequential baselines with the complete OMP sweep."""
    keys = matching_workloads(seq_raw, omp_raw)
    if not keys:
        raise ValueError("Experiments 1 and 2 have no exactly matching workloads")

    fig, axes = plt.subplots(1, len(keys), figsize=(6.4 * len(keys), 5), squeeze=False)
    for index, key in enumerate(keys):
        ax = axes[0, index]
        seq = select_workload(seq_raw, key)
        omp = select_workload(omp_raw, key)
        omp_summary = aggregate_interval(omp, ["num_threads"], "t_train").sort_values(
            "num_threads"
        )
        ax.errorbar(
            omp_summary["num_threads"],
            omp_summary["t_train"],
            yerr=interval_yerr(omp_summary, "t_train"),
            color=COLORS["omp"],
            marker="o",
            linewidth=2,
            capsize=3,
            label="OpenMP median ± IQR",
        )

        x_min = omp_summary["num_threads"].min()
        x_max = omp_summary["num_threads"].max()
        seq_summary = aggregate_interval(seq, ["variant"], "t_train")
        for _, row in seq_summary.iterrows():
            variant = row["variant"]
            label = "naive sequential" if variant == "naive" else "tiled sequential"
            color = COLORS.get(variant, "gray")
            ax.axhline(row["t_train"], color=color, linestyle="--", label=label)
            ax.fill_between(
                [x_min, x_max],
                row["t_train_q1"],
                row["t_train_q3"],
                color=color,
                alpha=0.10,
            )

        ax.set_yscale("log")
        set_power_of_two_axis(ax, omp_summary["num_threads"], "OpenMP threads")
        ax.set_ylabel("Training time (s, log scale)")
        ax.set_title(workload_title(key))
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle("Sequential Baselines versus OpenMP", fontsize=15)
    save_figure(fig, outdir, "seq_vs_omp.png")


def paired_scaling(data, worker_col: str):
    base = data[data[worker_col] == 1][["repetition", "t_train"]].rename(
        columns={"t_train": "t_base"}
    )
    paired = data.merge(base, on="repetition", how="inner")
    paired["speedup"] = paired["t_base"] / paired["t_train"]
    paired["efficiency"] = paired["speedup"] / paired[worker_col] * 100.0
    return paired


def plot_omp_vs_mpi(omp_raw, mpi_raw, outdir: str) -> None:
    """Compare self-speedup and efficiency on the common canonical workload."""
    keys = matching_workloads(omp_raw, mpi_raw)
    if len(keys) != 1:
        raise ValueError(
            "Experiments 2 and 3 must have exactly one matching strong-scaling workload"
        )
    key = keys[0]
    omp = paired_scaling(select_workload(omp_raw, key), "num_threads")
    mpi = paired_scaling(select_workload(mpi_raw, key), "num_ranks")

    fig, (ax_speed, ax_eff) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for name, data, worker_col, marker, color in [
        ("OpenMP", omp, "num_threads", "o", COLORS["omp"]),
        ("single-node MPI", mpi, "num_ranks", "s", COLORS["mpi"]),
    ]:
        speed = aggregate_interval(data, [worker_col], "speedup")
        efficiency = aggregate_interval(data, [worker_col], "efficiency")
        ax_speed.errorbar(
            speed[worker_col],
            speed["speedup"],
            yerr=interval_yerr(speed, "speedup"),
            color=color,
            marker=marker,
            linewidth=2,
            capsize=3,
            label=name,
        )
        ax_eff.errorbar(
            efficiency[worker_col],
            efficiency["efficiency"],
            yerr=interval_yerr(efficiency, "efficiency"),
            color=color,
            marker=marker,
            linewidth=2,
            capsize=3,
            label=name,
        )

    workers = sorted(omp["num_threads"].unique())
    ax_speed.plot(workers, workers, color="gray", linestyle=":", label="ideal linear")
    ax_speed.set_ylabel("Self-speedup")
    ax_speed.set_title("Speedup relative to each backend's 1-worker run")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=8)

    ax_eff.axhline(100.0, color="gray", linestyle=":")
    ax_eff.set_ylabel("Parallel efficiency (%)")
    ax_eff.set_ylim(bottom=0)
    ax_eff.set_title("Efficiency")
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend(fontsize=8)
    set_power_of_two_axis(ax_speed, workers, "Workers")
    set_power_of_two_axis(ax_eff, workers, "Workers")

    fig.suptitle(
        "OpenMP versus Single-Node MPI Strong Scaling\n" + workload_title(key),
        fontsize=15,
    )
    fig.text(
        0.5,
        0.005,
        "Systems comparison: MPI batch is local, so global batch and updates/epoch change with ranks.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(Path(outdir) / "omp_vs_mpi_strong.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("  -> omp_vs_mpi_strong.png")


def plot_tile_tuning(tile_raw, outdir: str) -> None:
    """Quantify the post-hoc benefit of the best tested tile over tile 64."""
    summary = aggregate_interval(tile_raw, ["num_ranks", "tile_size"], "t_train")
    ranks = sorted(summary["num_ranks"].unique())
    fixed_rows = []
    best_rows = []
    for ranks_value in ranks:
        subset = summary[summary["num_ranks"] == ranks_value]
        fixed = subset[subset["tile_size"] == 64]
        if fixed.empty:
            raise ValueError(f"tile sweep has no tile=64 baseline at {ranks_value} ranks")
        fixed_rows.append(fixed.iloc[0])
        best_rows.append(subset.loc[subset["t_train"].idxmin()])
    fixed = pd.DataFrame(fixed_rows)
    best = pd.DataFrame(best_rows)

    x = np.arange(len(ranks))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    fixed_bars = ax.bar(
        x - width / 2,
        fixed["t_train"],
        width,
        yerr=interval_yerr(fixed, "t_train"),
        capsize=4,
        color="#7f7f7f",
        label="fixed tile 64",
    )
    best_bars = ax.bar(
        x + width / 2,
        best["t_train"],
        width,
        yerr=interval_yerr(best, "t_train"),
        capsize=4,
        color=COLORS["mpi"],
        label="best tested tile",
    )
    for fixed_bar, best_bar, (_, fixed_row), (_, best_row) in zip(
        fixed_bars, best_bars, fixed.iterrows(), best.iterrows()
    ):
        improvement = (1.0 - best_row["t_train"] / fixed_row["t_train"]) * 100.0
        ax.annotate(
            f"tile {int(best_row['tile_size'])}\n{improvement:.1f}% faster",
            (best_bar.get_x() + best_bar.get_width() / 2, best_bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(x, [str(int(value)) for value in ranks])
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Training time (s, log scale)")
    ax.set_yscale("log")
    ax.set_title("MPI Tile Tuning Impact (Single Node)")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=8)
    save_figure(fig, outdir, "mpi_tile_tuning_impact.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--outdir", default="plots/cross_experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = Path(args.results_dir)
    paths = {
        "exp1": results / "exp1_seq_baseline.csv",
        "exp2": results / "exp2_omp_strong.csv",
        "exp3": results / "exp3_mpi_strong.csv",
        "exp6": results / "exp6_mpi_tile.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        print(f"Error: missing result files: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(1)

    apply_style()
    try:
        exp1 = load_csv(str(paths["exp1"]), BASE_REQUIRED + ["num_threads"])
        exp2 = load_csv(str(paths["exp2"]), BASE_REQUIRED + ["num_threads"])
        exp3 = load_csv(str(paths["exp3"]), BASE_REQUIRED + ["num_ranks"])
        exp6 = load_csv(
            str(paths["exp6"]), BASE_REQUIRED + ["num_ranks", "tile_size"]
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    os.makedirs(args.outdir, exist_ok=True)
    plot_seq_vs_omp(exp1, exp2, args.outdir)
    plot_omp_vs_mpi(exp2, exp3, args.outdir)
    plot_tile_tuning(exp6, args.outdir)
    print(f"Done. 3 cross-experiment plots generated in {args.outdir}/")


if __name__ == "__main__":
    main()
