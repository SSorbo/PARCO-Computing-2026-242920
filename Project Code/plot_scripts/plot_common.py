"""Shared helpers for the P2D report plotting scripts."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PLOT_STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (8, 5),
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}


def apply_style() -> None:
    """Apply one consistent, restrained report style."""
    plt.rcParams.update(PLOT_STYLE)


def load_csv(csv_path: str, required: list[str]) -> pd.DataFrame:
    """Load a result CSV, validate columns, and reject non-finite timings."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {', '.join(missing)}")

    numeric = set(required) - {"experiment", "backend", "variant"}
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.empty:
        raise ValueError(f"{csv_path} contains no result rows")
    if not (df["t_train"] > 0).all():
        raise ValueError(f"{csv_path} contains non-positive training times")
    return df


def aggregate_interval(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Return median, quartiles, and repetition count for one measurement."""
    name = prefix or value_col
    return (
        df.groupby(group_cols, dropna=False)[value_col]
        .agg(
            **{
                name: "median",
                f"{name}_q1": lambda values: values.quantile(0.25),
                f"{name}_q3": lambda values: values.quantile(0.75),
                f"{name}_count": "count",
            }
        )
        .reset_index()
    )


def interval_yerr(data: pd.DataFrame, value: str):
    """Build asymmetric Matplotlib error bars from q1/median/q3 columns."""
    return [
        (data[value] - data[f"{value}_q1"]).clip(lower=0).to_numpy(),
        (data[f"{value}_q3"] - data[value]).clip(lower=0).to_numpy(),
    ]


def config_columns(df: pd.DataFrame, include_batch: bool = True) -> list[str]:
    """Return the columns that identify a comparable fixed workload."""
    columns = [
        "experiment",
        "samples",
        "input_size",
        "hidden_size",
        "output_size",
        "epochs",
    ]
    if include_batch:
        columns.append("batch_size")
    return [column for column in columns if column in df.columns]


def iter_configs(df: pd.DataFrame, columns: list[str]):
    """Yield configuration dictionaries and matching data in stable order."""
    groups = df.groupby(columns, sort=True, dropna=False)
    for key, subset in groups:
        if not isinstance(key, tuple):
            key = (key,)
        yield dict(zip(columns, key)), subset.copy()


def short_experiment_name(experiment: str) -> str:
    """Convert an experiment identifier into a compact facet label."""
    if "large_input" in experiment:
        return "Large input"
    if "canonical" in experiment:
        return "Canonical"
    if experiment in {"exp3_mpi_strong", "exp6_mpi_tile"}:
        return "Canonical"
    return experiment.replace("_", " ")


def config_title(config: dict, include_batch: bool = False) -> str:
    """Build a compact two-line workload title."""
    name = short_experiment_name(str(config.get("experiment", "Configuration")))
    shape = (
        f"N={int(config['samples'])}, "
        f"{int(config['input_size'])}\u2192{int(config['hidden_size'])}"
        f"\u2192{int(config['output_size'])}, "
        f"{int(config['epochs'])} epochs"
    )
    if include_batch and "batch_size" in config:
        shape += f", batch={int(config['batch_size'])}"
    return f"{name}\n{shape}"


def set_power_of_two_axis(ax, values, label: str) -> None:
    """Use a readable base-two axis for worker and power-of-two sweeps."""
    ticks = sorted({int(value) for value in values})
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(label)


def save_figure(fig, outdir: str, filename: str) -> None:
    """Save and close a figure."""
    os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(Path(outdir) / filename)
    plt.close(fig)
    print(f"  -> {filename}")


def remove_outputs(outdir: str, filenames: list[str]) -> None:
    """Remove obsolete outputs so skipped plots cannot survive from old runs."""
    for filename in filenames:
        path = Path(outdir) / filename
        if path.exists():
            path.unlink()
            print(f"  removed obsolete {filename}")
