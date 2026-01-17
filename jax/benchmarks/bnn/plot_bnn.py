# plot_bnn.py --------------------------------------------------------------
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


METRICS_DIR = Path("metrics") / "bnn"
FIG_D = Path("figures") / "bnn"
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

palette = sns.color_palette("tab10", n_colors=len(plot_methods))
COLOR_MAP = {m: c for m, c in zip(plot_methods, palette)}


def _label(method):
    return METHOD_LABELS.get(method, method)


def _load_datasets():
    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Missing metrics directory: {METRICS_DIR}")
    return sorted(METRICS_DIR.glob("*.csv"))


def _latest_per_run(df):
    if "seed" not in df.columns:
        return df.sort_values("iter").groupby("method").tail(1).copy()
    df_last = df.sort_values("iter").groupby(["method", "seed"]).tail(1)
    if "is_best" in df.columns:
        df_best = df[df["is_best"] == 1]
        if not df_best.empty:
            best_idx = df_best.set_index(["method", "seed"])
            last_idx = df_last.set_index(["method", "seed"])
            return best_idx.combine_first(last_idx).reset_index()
    return df_last.reset_index(drop=True)


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
    ax.set_xticklabels([_label(m) for m in plot_methods], rotation=30, ha="right")
    ax.set_xlabel("")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"saved → {fname}")
    plt.close()


def _summarize_latest(latest):
    metrics = ["acc", "nll", "ece", "ess"]
    rows = []
    for name, sub in latest.groupby("method"):
        row = {"method": _label(name), "n_runs": int(sub.shape[0])}
        for metric in metrics:
            if metric not in sub.columns:
                continue
            vals = sub[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_summary_table(summary, fname):
    if summary.empty:
        return
    cols = ["method", "n_runs", "acc_mean", "acc_std", "nll_mean", "nll_std", "ece_mean", "ece_std", "ess_mean", "ess_std"]
    cols = [c for c in cols if c in summary.columns]
    table_df = summary[cols].copy()
    for col in table_df.columns:
        if col.endswith("_mean") or col.endswith("_std"):
            table_df[col] = table_df[col].map(lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(12, 2 + 0.3 * len(table_df)))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"saved → {fname}")
    plt.close(fig)


def _plot_metric_means_panels(latest, fname):
    if latest.empty:
        return
    metrics = [("nll", "NLL"), ("ece", "ECE"), ("acc", "Accuracy"), ("ess", "ESS")]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharey=False)
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
    plt.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"saved → {fname}")
    plt.close(fig)


def _plot_speed_accuracy_scatter(latest, fname):
    if latest.empty or "wall_s" not in latest.columns:
        return
    metrics = [("nll", "NLL"), ("ece", "ECE")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=False, sharey=False)
    methods = [m for m in plot_methods if m in latest["method"].unique()]

    for ax, (metric, ylabel) in zip(axes, metrics):
        for m in methods:
            sub = latest[latest["method"] == m]
            x_vals = sub["wall_s"].to_numpy()
            y_vals = sub[metric].to_numpy()
            if not np.isfinite(x_vals).any() or not np.isfinite(y_vals).any():
                continue
            x_mean = float(np.nanmean(x_vals))
            y_mean = float(np.nanmean(y_vals))
            x_std = float(np.nanstd(x_vals))
            y_std = float(np.nanstd(y_vals))
            ax.errorbar(
                x_mean,
                y_mean,
                xerr=x_std,
                yerr=y_std,
                fmt="o",
                color=COLOR_MAP.get(m),
                ecolor=COLOR_MAP.get(m),
                elinewidth=1.2,
                capsize=4,
                alpha=0.9,
                label=_label(m),
            )
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"saved → {fname}")
    plt.close(fig)


def _plot_dataset(csv_path):
    dataset = csv_path.stem
    out_dir = FIG_D / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[df["method"].isin(plot_methods)]

    def _fname(base):
        return out_dir / f"{dataset}_{base}.png"

    latest = _latest_per_run(df)
    _plot_bars(latest, "acc", "Accuracy", _fname("acc_bar"))
    _plot_bars(latest, "nll", "NLL", _fname("nll_bar"))
    _plot_bars(latest, "ece", "ECE", _fname("ece_bar"))

    _plot_metric_means_panels(latest, _fname("summary_metric_panels"))
    summary = _summarize_latest(latest)
    summary.to_csv(out_dir / f"{dataset}_summary.csv", index=False)
    _plot_summary_table(summary, _fname("summary_table"))
    _plot_speed_accuracy_scatter(latest, _fname("speed_accuracy_scatter"))


def main():
    csv_files = _load_datasets()
    if not csv_files:
        print(f"No CSV files found in {METRICS_DIR}")
        return
    for csv_path in csv_files:
        _plot_dataset(csv_path)


if __name__ == "__main__":
    main()
