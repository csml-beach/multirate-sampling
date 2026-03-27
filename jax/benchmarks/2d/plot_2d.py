# plot_2d.py ---------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 20,
    }
)

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))


METRICS_DIR = Path("metrics") / "2d"
FIG_D = Path("figures") / "2d"
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
def _load_targets():
    if not METRICS_DIR.exists():
        raise FileNotFoundError(f"Missing metrics directory: {METRICS_DIR}")
    return sorted(METRICS_DIR.glob("*.csv"))

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


def _prepare_line_df(df_in):
    df_line = df_in.copy()
    if "is_best" in df_line.columns:
        df_line = df_line[df_line["is_best"] != 1]
    if "seed" in df_line.columns:
        df_line = df_line.groupby(["method", "iter"], as_index=False).mean(numeric_only=True)
    return df_line


def _latest_per_run(df_in):
    if "seed" not in df_in.columns:
        return df_in.sort_values("iter").groupby("method").tail(1).copy()
    df_last = df_in.sort_values("iter").groupby(["method", "seed"]).tail(1)
    if "is_best" in df_in.columns:
        df_best = df_in[df_in["is_best"] == 1]
        if not df_best.empty:
            best_idx = df_best.set_index(["method", "seed"])
            last_idx = df_last.set_index(["method", "seed"])
            return best_idx.combine_first(last_idx).reset_index()
    return df_last.reset_index(drop=True)


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


def _plot_mu_cov_final_panels(df_raw, out_path):
    if "mu_err" not in df_raw.columns or "cov_err" not in df_raw.columns:
        return
    latest = _latest_per_run(df_raw)
    latest = latest[np.isfinite(latest["mu_err"]) & np.isfinite(latest["cov_err"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)

    latest["size_ess"] = _size_from_values(latest["ess"].values)
    for name, sub in latest.groupby("method"):
        axes[0].scatter(
            sub["mu_err"].values,
            sub["cov_err"].values,
            color=COLOR_MAP.get(name),
            marker="o",
            s=sub["size_ess"].values,
            alpha=0.9,
            label=_label(name),
            edgecolors="none",
        )
    axes[0].set_title("Final: size = ESS")

    latest["size_wall"] = _size_from_values(latest["wall_s"].values)
    for name, sub in latest.groupby("method"):
        axes[1].scatter(
            sub["mu_err"].values,
            sub["cov_err"].values,
            color=COLOR_MAP.get(name),
            marker="o",
            s=sub["size_wall"].values,
            alpha=0.9,
            label=_label(name),
            edgecolors="none",
        )
    axes[1].set_title("Final: size = wall time")

    if np.all(latest["mu_err"].values > 0) and np.all(latest["cov_err"].values > 0):
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
    for ax in axes:
        ax.set_xlabel(r"$\Vert\hat\mu-\mu\Vert_2$")
        ax.set_ylabel(r"$\Vert\hat\Sigma-\Sigma\Vert_F$")

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
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved → {out_path}")
    plt.close(fig)


def _plot_dual_x(df_line, metric, ylabel, fname, logy=True, smooth=True, window=7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    x_col, x_label = _resolve_xaxis(df_line, False)
    left_vals = df_line[metric].to_numpy()
    logy_left = logy and np.all(left_vals > 0)
    if logy and not logy_left:
        print(f"note: disabling log-y for {metric} (non-positive values)")
    for name, sub in df_line.groupby("method"):
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

    if "kernel_evals" in df_line.columns:
        right_vals = df_line.loc[df_line["kernel_evals"] > 0, metric].to_numpy()
        logy_right = logy and right_vals.size > 0 and np.all(right_vals > 0)
        if logy and not logy_right and right_vals.size > 0:
            print(f"note: disabling log-y for {metric} (kernel-evals) (non-positive values)")
        for name, sub in df_line.groupby("method"):
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
    fp = Path(fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)


def _plot_metric_means_panels(latest, fname):
    if latest.empty:
        return
    metrics = [
        ("ksd", "KSD"),
        ("mean_logp", "Mean log-prob"),
        ("ess", "ESS"),
        ("wall_s", "Wall time (s)"),
    ]
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
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(np.nanmean(vals)))
                stds.append(float(np.nanstd(vals)))
        for idx, m in enumerate(methods):
            if not np.isfinite(means[idx]):
                continue
            ax.errorbar(
                x[idx],
                means[idx],
                yerr=stds[idx],
                fmt="o",
                color="black",
                ecolor="black",
                elinewidth=1.2,
                capsize=4,
            )
            ax.scatter(x[idx], means[idx], color=COLOR_MAP.get(m), s=60, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in methods], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        finite_means = np.array([v for v in means if np.isfinite(v)])
        if metric == "ksd" and finite_means.size > 0 and np.all(finite_means > 0):
            ax.set_yscale("log")

    plt.tight_layout()
    fp = Path(fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)


def _plot_target(csv_path):
    target = csv_path.stem
    out_dir = FIG_D / target
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw[df_raw["method"].isin(plot_methods)]
    df_line = _prepare_line_df(df_raw)

    def _fname(base):
        return f"{target}_{base}.png"

    _plot_dual_x(df_line, "mu_err", r"$\Vert\hat\mu-\mu\Vert_2$", out_dir / _fname("mu_error_dual_axis"))
    _plot_dual_x(df_line, "cov_err", r"$\Vert\hat\Sigma-\Sigma\Vert_F$", out_dir / _fname("cov_error_dual_axis"))

    if "ksd" in df_line.columns:
        _plot_dual_x(df_line, "ksd", "KSD", out_dir / _fname("ksd_dual_axis"), logy=True)

    if "grid_l1" in df_line.columns:
        _plot_dual_x(df_line, "grid_l1", "Grid L1", out_dir / _fname("grid_l1_dual_axis"), logy=False)

    if "mean_logp" in df_line.columns:
        _plot_dual_x(df_line, "mean_logp", "Mean log-prob", out_dir / _fname("mean_logp_dual_axis"), logy=False)

    _plot_mu_cov_final_panels(
        df_raw,
        out_dir / _fname("pareto_mu_cov_final"),
    )

    latest = _latest_per_run(df_raw)
    _plot_metric_means_panels(latest, out_dir / _fname("summary_metric_panels"))

    plt.figure(figsize=(9.5, 4.5))
    ax = sns.barplot(
        data=latest,
        x="method",
        y="ess",
        hue="method",
        palette="Set2",
        legend=False,
        order=plot_methods,
    )
    ax.set_xticks(range(len(plot_methods)))
    ax.set_xticklabels([_label(m) for m in plot_methods], rotation=30, ha="right")
    ax.set_xlabel("")
    plt.ylabel("ESS (chain dim-0)")
    plt.tight_layout()
    fp = out_dir / _fname("ess_bar")
    plt.savefig(fp, dpi=150)
    print(f"saved → {fp}")
    plt.close()

    latest = latest.copy()
    _grad_col = "grad_evals" if "grad_evals" in latest.columns else "iter"
    latest["ess_per_grad"] = latest["ess"] / latest[_grad_col].replace(0, np.nan)
    plt.figure(figsize=(9.5, 4.5))
    ax = sns.barplot(
        data=latest,
        x="method",
        y="ess_per_grad",
        hue="method",
        palette="Set2",
        legend=False,
        order=plot_methods,
    )
    ax.set_xticks(range(len(plot_methods)))
    ax.set_xticklabels([_label(m) for m in plot_methods], rotation=30, ha="right")
    ax.set_xlabel("")
    plt.ylabel("ESS / grad eval")
    plt.tight_layout()
    fp = out_dir / _fname("ess_per_grad_bar")
    plt.savefig(fp, dpi=150)
    print(f"saved → {fp}")
    plt.close()


def main():
    csv_files = _load_targets()
    if not csv_files:
        print(f"No CSV files found in {METRICS_DIR}")
        return
    for csv_path in csv_files:
        _plot_target(csv_path)


if __name__ == "__main__":
    main()
