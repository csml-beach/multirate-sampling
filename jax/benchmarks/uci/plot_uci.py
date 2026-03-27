# plot_uci.py --------------------------------------------------------------
from pathlib import Path
import os
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


METRICS_DIR = Path("metrics") / "uci"
FIG_D = Path("figures") / "uci"
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


def _load_datasets():
    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Missing metrics directory: {METRICS_DIR}")
    return sorted(METRICS_DIR.glob("*.csv"))


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


def _size_from_values(values, min_size=60, max_size=240):
    v = np.asarray(values)
    if v.size == 0:
        return v
    v_min = np.nanmin(v)
    v_max = np.nanmax(v)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return np.full_like(v, 120.0, dtype=float)
    return np.interp(v, (v_min, v_max), (min_size, max_size))


def _get_run_cols(df):
    return [col for col in ("seed", "split", "run_id") if col in df.columns]


def _latest_per_run(df):
    run_cols = _get_run_cols(df)
    if run_cols:
        if "is_best" in df.columns:
            df_last = df.sort_values(run_cols + ["iter"]).groupby(["method"] + run_cols).tail(1)
            df_best = df[df["is_best"] == 1]
            if not df_best.empty:
                best_idx = df_best.set_index(["method"] + run_cols)
                last_idx = df_last.set_index(["method"] + run_cols)
                return best_idx.combine_first(last_idx).reset_index()
        return df.groupby(["method"] + run_cols, sort=False).tail(1).copy()
    if "is_best" in df.columns:
        df_last = df.sort_values("iter").groupby("method").tail(1)
        df_best = df[df["is_best"] == 1]
        if not df_best.empty:
            best_idx = df_best.set_index(["method"])
            last_idx = df_last.set_index(["method"])
            return best_idx.combine_first(last_idx).reset_index()
    return df.groupby("method", sort=False).tail(1).copy()


def _plot_dual_x(df, metric, ylabel, fname, logy=False, smooth=True, window=7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    x_col, x_label = _resolve_xaxis(df, False)
    left_vals = df[metric].to_numpy()
    logy_left = logy and np.all(left_vals > 0)
    if logy and not logy_left:
        print(f"note: disabling log-y for {metric} (non-positive values)")
    for name, sub in df.groupby("method"):
        x, y = sub[x_col].values, sub[metric].values
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
    if logy_left:
        axes[0].set_yscale("log")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(ylabel)

    if "kernel_evals" in df.columns:
        right_vals = df.loc[df["kernel_evals"] > 0, metric].to_numpy()
        logy_right = logy and right_vals.size > 0 and np.all(right_vals > 0)
        if logy and not logy_right and right_vals.size > 0:
            print(f"note: disabling log-y for {metric} (kernel-evals) (non-positive values)")
        for name, sub in df.groupby("method"):
            sub = sub[sub["kernel_evals"] > 0]
            if sub.empty:
                continue
            x, y = sub["kernel_evals"].values, sub[metric].values
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
        if logy_right:
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


def _summarize_latest(latest):
    metrics = ["acc", "nll", "ece"]
    rows = []
    for name, sub in latest.groupby("method"):
        row = {"method": _label(name), "n_runs": int(sub.shape[0])}
        for metric in metrics:
            vals = sub[metric].to_numpy()
            row[f"{metric}_mean"] = float(np.nanmean(vals))
            row[f"{metric}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_summary_table(summary, fname):
    if summary.empty:
        return
    cols = ["method", "n_runs", "acc_mean", "acc_std", "nll_mean", "nll_std", "ece_mean", "ece_std"]
    table_df = summary[cols].copy()
    table_df["acc_mean"] = table_df["acc_mean"].map(lambda v: f"{v:.3f}")
    table_df["acc_std"] = table_df["acc_std"].map(lambda v: f"{v:.3f}")
    table_df["nll_mean"] = table_df["nll_mean"].map(lambda v: f"{v:.3f}")
    table_df["nll_std"] = table_df["nll_std"].map(lambda v: f"{v:.3f}")
    table_df["ece_mean"] = table_df["ece_mean"].map(lambda v: f"{v:.3f}")
    table_df["ece_std"] = table_df["ece_std"].map(lambda v: f"{v:.3f}")

    fig, ax = plt.subplots(figsize=(11, 2 + 0.3 * len(table_df)))
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
            ax.scatter(
                x[idx],
                means[idx],
                color=COLOR_MAP.get(m),
                s=60,
                zorder=3,
            )
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


def _plot_nll_ece_final_panels(df, fname):
    if "nll" not in df.columns or "ece" not in df.columns:
        return
    latest = _latest_per_run(df)
    latest = latest[np.isfinite(latest["nll"]) & np.isfinite(latest["ece"])]
    run_cols = _get_run_cols(latest)
    multi_runs = bool(run_cols) and latest.groupby("method").size().max() > 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)

    for name, sub in latest.groupby("method"):
        if sub.empty or "wall_s" not in sub.columns:
            continue
        size_vals = _size_from_values(sub["wall_s"].values)
        if multi_runs:
            axes[0].scatter(
                sub["nll"].values,
                sub["ece"].values,
                color=COLOR_MAP.get(name),
                marker="o",
                s=size_vals,
                alpha=0.35,
                label=None,
                edgecolors="none",
            )
            med_x = float(np.nanmedian(sub["nll"].values))
            med_y = float(np.nanmedian(sub["ece"].values))
            med_size = _size_from_values([np.nanmedian(sub["wall_s"].values)])[0]
            axes[0].scatter(
                [med_x],
                [med_y],
                color=COLOR_MAP.get(name),
                marker="o",
                s=med_size * 1.6,
                alpha=0.95,
                label=_label(name),
                edgecolors="black",
                linewidths=0.8,
            )
        else:
            axes[0].scatter(
                sub["nll"].values,
                sub["ece"].values,
                color=COLOR_MAP.get(name),
                marker="o",
                s=size_vals,
                alpha=0.9,
                label=_label(name),
                edgecolors="none",
            )
    axes[0].set_title("Final (runs): size = wall time" if multi_runs else "Final: size = wall time")

    if "ess" in latest.columns:
        for name, sub in latest.groupby("method"):
            if sub.empty:
                continue
            size_vals = _size_from_values(sub["ess"].values)
            if multi_runs:
                axes[1].scatter(
                    sub["nll"].values,
                    sub["ece"].values,
                    color=COLOR_MAP.get(name),
                    marker="o",
                    s=size_vals,
                    alpha=0.35,
                    label=None,
                    edgecolors="none",
                )
                med_x = float(np.nanmedian(sub["nll"].values))
                med_y = float(np.nanmedian(sub["ece"].values))
                med_size = _size_from_values([np.nanmedian(sub["ess"].values)])[0]
                axes[1].scatter(
                    [med_x],
                    [med_y],
                    color=COLOR_MAP.get(name),
                    marker="o",
                    s=med_size * 1.6,
                    alpha=0.95,
                    label=_label(name),
                    edgecolors="black",
                    linewidths=0.8,
                )
            else:
                axes[1].scatter(
                    sub["nll"].values,
                    sub["ece"].values,
                    color=COLOR_MAP.get(name),
                    marker="o",
                    s=size_vals,
                    alpha=0.9,
                    label=_label(name),
                    edgecolors="none",
                )
        axes[1].set_title("Final (runs): size = ESS" if multi_runs else "Final: size = ESS")
    else:
        axes[1].axis("off")

    for ax in axes:
        ax.set_xlabel("NLL")
        ax.set_ylabel("ECE")

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=False,
    )
    for h in legend.legend_handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([60])
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
