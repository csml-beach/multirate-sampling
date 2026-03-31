# plot_mixture2d.py --------------------------------------------------------
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 20,
    }
)

METRICS_DIR = Path("metrics") / "mixture2d"
FIG_D = Path("figures") / "mixture2d"
FIG_D.mkdir(parents=True, exist_ok=True)

plot_methods = [
    "vanilla_svgd",
    "strang_svgd",
    "multirate_svgd",
    "adaptive_multirate_svgd",
    "sgld",
    "sghmc",
]

METHOD_LABELS = {
    "adaptive_multirate_svgd": "Adapt-MR-SVGD",
    "multirate_svgd": "MR-SVGD",
    "vanilla_svgd": "SVGD",
    "strang_svgd": "Strang-SVGD",
    "sgld": "SGLD",
    "sghmc": "SGHMC",
}

LINE_STYLES = {
    "sgld": "--",
    "sghmc": "--",
    "vanilla_svgd": "-.",
    "strang_svgd": "-.",
    "multirate_svgd": "-",
    "adaptive_multirate_svgd": "-",
}

palette = sns.color_palette("tab10", n_colors=len(plot_methods))
COLOR_MAP = {m: c for m, c in zip(plot_methods, palette)}


def _label(method):
    return METHOD_LABELS.get(method, method)


def _load_datasets(dataset_names=None):
    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Missing metrics directory: {METRICS_DIR}")
    csvs = sorted(METRICS_DIR.glob("*.csv"))
    if dataset_names is None:
        return csvs
    wanted = set(dataset_names)
    return [csv_path for csv_path in csvs if csv_path.stem in wanted]


def _resolve_xaxis(df, use_kernel_evals):
    if use_kernel_evals and "kernel_evals" in df.columns:
        return "kernel_evals", "Kernel evaluations"
    if "grad_evals" in df.columns:
        return "grad_evals", "Gradient evaluations"
    return "iter", "Iterations"


def smooth_data(x, y, window=7):
    if len(y) < window:
        return x, y
    y_padded = np.concatenate([np.repeat(y[0], window // 2), y, np.repeat(y[-1], window // 2)])
    y_smooth = np.convolve(y_padded, np.ones(window) / window, mode="valid")
    return x, y_smooth


def _plot_dual_x(df, metric, ylabel, fname, logy=False, smooth=True, window=7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    x_col, x_label = _resolve_xaxis(df, False)
    for name, sub in df.groupby("method"):
        curve = sub.groupby(x_col, as_index=False)[metric].mean().sort_values(x_col)
        x, y = curve[x_col].values, curve[metric].values
        if smooth:
            x, y = smooth_data(x, y, window)
        axes[0].plot(
            x,
            y,
            label=_label(name),
            linewidth=2,
            linestyle=LINE_STYLES.get(name, "-"),
            color=COLOR_MAP.get(name),
        )
    axes[0].set_xscale("log")
    if logy:
        axes[0].set_yscale("log")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(ylabel)

    if "kernel_evals" in df.columns:
        for name, sub in df.groupby("method"):
            sub = sub[sub["kernel_evals"] > 0]
            if sub.empty:
                continue
            curve = sub.groupby("kernel_evals", as_index=False)[metric].mean().sort_values("kernel_evals")
            x, y = curve["kernel_evals"].values, curve[metric].values
            if smooth:
                x, y = smooth_data(x, y, window)
            axes[1].plot(
                x,
                y,
                label=_label(name),
                linewidth=2,
                linestyle=LINE_STYLES.get(name, "-"),
                color=COLOR_MAP.get(name),
            )
        axes[1].set_xscale("log")
        if logy:
            axes[1].set_yscale("log")
        axes[1].set_xlabel("Kernel evaluations")
    else:
        axes[1].axis("off")
        axes[1].set_xlabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"saved → {fname}")
    plt.close(fig)


def _plot_bars(latest, metric, ylabel, fname):
    plt.figure(figsize=(9.5, 4.5))
    ax = sns.barplot(
        data=latest,
        x="method",
        y=metric,
        hue="method",
        palette="Set2",
        legend=False,
        order=plot_methods,
    )
    ax.set_xticks(range(len(plot_methods)))
    ax.set_xticklabels([_label(m) for m in plot_methods], rotation=30, ha="right")
    ax.set_xlabel("")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"saved → {fname}")
    plt.close()


def _plot_summary_panels(latest, fname):
    if latest.empty:
        return
    metrics = [
        ("coverage", "Mode coverage"),
        ("entropy", "Mode entropy"),
        ("imbalance", "Mode imbalance"),
        ("ksd", "KSD"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharey=False)
    axes = axes.flatten()
    methods = [m for m in plot_methods if m in latest["method"].unique()]
    x = np.arange(len(methods))

    for ax, (metric, ylabel) in zip(axes, metrics):
        if metric not in latest.columns:
            ax.axis("off")
            continue
        means = []
        stds = []
        for m in methods:
            vals = latest.loc[latest["method"] == m, metric].to_numpy()
            means.append(np.nanmean(vals))
            stds.append(np.nanstd(vals))
        ax.errorbar(
            x,
            means,
            yerr=stds,
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.2,
            capsize=4,
        )
        for idx, m in enumerate(methods):
            ax.scatter(x[idx], means[idx], color=COLOR_MAP.get(m), s=60, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in methods], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if metric == "ksd" and np.all(np.array(means) > 0):
            ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"saved → {fname}")
    plt.close(fig)


def _plot_dataset(csv_path):
    target = csv_path.stem
    out_dir = FIG_D / target
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["method"].isin(plot_methods)]

    def _fname(base):
        return out_dir / f"{target}_{base}.png"

    _plot_dual_x(df, "coverage", "Mode coverage", _fname("coverage_dual_axis"), logy=False)
    _plot_dual_x(df, "entropy", "Mode entropy", _fname("entropy_dual_axis"), logy=False)
    if "grid_l1" in df.columns:
        _plot_dual_x(df, "grid_l1", "Grid L1", _fname("grid_l1_dual_axis"), logy=True)
    if "ksd" in df.columns:
        _plot_dual_x(df, "ksd", "KSD", _fname("ksd_dual_axis"), logy=True)

    if "seed" in df.columns:
        latest = df.sort_values("iter").groupby(["method", "seed"]).tail(1)
    else:
        latest = df.sort_values("iter").groupby("method").tail(1)
    _plot_bars(latest, "coverage", "Mode coverage", _fname("coverage_bar"))
    _plot_bars(latest, "entropy", "Mode entropy", _fname("entropy_bar"))
    _plot_summary_panels(latest, _fname("summary_metric_panels"))


def parse_args():
    parser = argparse.ArgumentParser(description="Plot mixture benchmark summaries.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset stems to plot, e.g. mix8 mix8_imq.")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_files = _load_datasets(args.datasets)
    if not csv_files:
        requested = args.datasets if args.datasets else ['<all>']
        print(f"No CSV files found in {METRICS_DIR} for {requested}")
        return
    for csv_path in csv_files:
        _plot_dataset(csv_path)


if __name__ == "__main__":
    main()
