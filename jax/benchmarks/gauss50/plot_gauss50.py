# plot_gauss50.py -----------------------------------------------------------
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os, textwrap
import numpy as np
import sys
from pathlib import Path

matplotlib.rcParams.update(
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

CSV   = os.path.join("metrics", "50d", "metrics_gauss50.csv")
FIG_D = os.path.join("figures", "50d")
os.makedirs(FIG_D, exist_ok=True)

df_raw = pd.read_csv(CSV)

# Always render to PNGs only (non-interactive).

# Methods to plot
plot_methods = [
    "vanilla_svgd",
    "strang_svgd",
    "multirate_svgd",
    "adaptive_multirate_svgd",
    "sgld",
    "sghmc",
]
df_raw = df_raw[df_raw["method"].isin(plot_methods)]

METHOD_LABELS = {
    "adaptive_multirate_svgd": "Adapt-MR-SVGD",
    "multirate_svgd": "MR-SVGD",
    "vanilla_svgd": "SVGD",
    "strang_svgd": "Strang-SVGD",
    "sgld": "SGLD",
    "sghmc": "SGHMC",
}

# Line styles by method family
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


df_line = _prepare_line_df(df_raw)


def _latest_per_run(df_in):
    if "seed" not in df_in.columns:
        return df_in.sort_values("iter").groupby("method").tail(1).copy()
    df_last = df_in.sort_values("iter").groupby(["method", "seed"]).tail(1)
    if "is_best" in df_in.columns:
        df_best = df_in[df_in["is_best"] == 1]
        if not df_best.empty:
            best_idx = df_best.set_index(["method", "seed"])
            last_idx = df_last.set_index(["method", "seed"])
            combined = best_idx.combine_first(last_idx)
            return combined.reset_index()
    return df_last.reset_index(drop=True)

def _resolve_xaxis(df_in, use_kernel_evals):
    if use_kernel_evals and "kernel_evals" in df_in.columns:
        return "kernel_evals", "Kernel evaluations"
    if "grad_evals" in df_in.columns:
        return "grad_evals", "Gradient evaluations"
    return "iter", "Iterations"

# ------------------------------------------------------------------ helper
def smooth_data(x, y, window=7):
    """Apply simple moving average smoothing."""
    if len(y) < window:
        return x, y
    
    # Pad the data to handle edges
    y_padded = np.concatenate([np.repeat(y[0], window//2), y, np.repeat(y[-1], window//2)])
    y_smooth = np.convolve(y_padded, np.ones(window)/window, mode='valid')
    return x, y_smooth

def _plot(metric, ylabel, fname, logy=True, smooth=True, window=7, use_kernel_evals=False):
    x_col, x_label = _resolve_xaxis(df_line, use_kernel_evals)
    plt.figure(figsize=(9, 4))
    for name, sub in df_line.groupby("method"):
        x, y = sub[x_col].values, sub[metric].values
        if smooth:
            x, y = smooth_data(x, y, window)
        plt.plot(
            x, y,
            label=_label(name),
            linewidth=2,
            linestyle=LINE_STYLES.get(name, "-"),
            color=COLOR_MAP.get(name),
        )
    plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    title_suffix = " (smoothed)" if smooth else ""
    plt.title(f"{ylabel} vs {x_label.lower()} (50-dim Gaussian){title_suffix}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    fp = os.path.join(FIG_D, fname)
    plt.savefig(fp, dpi=150)
    print(f"saved → {fp}")
    plt.close()

def _plot_dual_x(metric, ylabel, fname, logy=True, smooth=True, window=7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    x_col, x_label = _resolve_xaxis(df_line, False)
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
    if logy:
        axes[0].set_yscale("log")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(ylabel)

    if "kernel_evals" in df_line.columns:
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
    fp = os.path.join(FIG_D, fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)

# 1) mean-error curve (dual-axis)
_plot_dual_x("mu_err", r"$\Vert\hat\mu\Vert_2$", "mu_error_dual_axis.png")

# 2) covariance Frobenius error curve (dual-axis)
_plot_dual_x("cov_err", r"$\Vert\hat\Sigma-\Sigma\Vert_F$", "cov_error_dual_axis.png")

# 3) KSD curve (if available, dual-axis)
if "ksd" in df_line.columns:
    _plot_dual_x("ksd", "KSD", "ksd_dual_axis.png", logy=True)

# 3b) wall-time overview (stacked panels)
def _plot_walltime_overview(metrics, ylabels, fname, logy_flags=None, smooth=True, window=7):
    if "wall_s" not in df_line.columns:
        return
    nrows = len(metrics)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 10), sharex=True)
    if nrows == 1:
        axes = [axes]
    for idx, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[idx]
        for name, sub in df_line.groupby("method"):
            x, y = sub["wall_s"].values, sub[metric].values
            if smooth:
                x, y = smooth_data(x, y, window)
            ax.plot(
                x,
                y,
                label=_label(name),
                linewidth=2,
                linestyle=LINE_STYLES.get(name, "-"),
                color=COLOR_MAP.get(name),
            )
        ax.set_xscale("log")
        if logy_flags is None or logy_flags[idx]:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Wall time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(FIG_D, fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)

_plot_walltime_overview(
    ["mu_err", "cov_err", "ksd"],
    [r"$\Vert\hat\mu\Vert_2$", r"$\Vert\hat\Sigma-\Sigma\Vert_F$", "KSD"],
    "walltime_overview.png",
    logy_flags=[True, True, True],
)

def _size_from_values(values, min_size=60, max_size=240):
    v = np.asarray(values)
    if v.size == 0:
        return v
    v_min = np.nanmin(v)
    v_max = np.nanmax(v)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return np.full_like(v, 120.0, dtype=float)
    return np.interp(v, (v_min, v_max), (min_size, max_size))


def _plot_mu_cov_final_panels(fname):
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
        ax.set_xlabel(r"$\Vert\hat\mu\Vert_2$")
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
    fp = os.path.join(FIG_D, fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)

_plot_mu_cov_final_panels("pareto_mu_cov_final.png")

def _plot_mu_cov_final_error_panels(fname):
    if "mu_err" not in df_raw.columns or "cov_err" not in df_raw.columns:
        return
    latest = _latest_per_run(df_raw)
    latest = latest[np.isfinite(latest["mu_err"]) & np.isfinite(latest["cov_err"])]
    if latest.empty:
        return

    methods = [m for m in plot_methods if m in latest["method"].unique()]
    if not methods:
        return

    means_mu = []
    stds_mu = []
    means_cov = []
    stds_cov = []
    means_ess = []
    means_wall = []
    for m in methods:
        sub = latest[latest["method"] == m]
        mu_vals = sub["mu_err"].to_numpy()
        cov_vals = sub["cov_err"].to_numpy()
        means_mu.append(float(np.nanmean(mu_vals)))
        stds_mu.append(float(np.nanstd(mu_vals)))
        means_cov.append(float(np.nanmean(cov_vals)))
        stds_cov.append(float(np.nanstd(cov_vals)))
        if "ess" in sub.columns:
            means_ess.append(float(np.nanmean(sub["ess"].to_numpy())))
        else:
            means_ess.append(np.nan)
        if "wall_s" in sub.columns:
            means_wall.append(float(np.nanmean(sub["wall_s"].to_numpy())))
        else:
            means_wall.append(np.nan)

    size_ess = _size_from_values(np.asarray(means_ess))
    size_wall = _size_from_values(np.asarray(means_wall))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for idx, m in enumerate(methods):
        color = COLOR_MAP.get(m)
        axes[0].errorbar(
            means_mu[idx],
            means_cov[idx],
            xerr=stds_mu[idx],
            yerr=stds_cov[idx],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=4,
            alpha=0.9,
        )
        axes[0].scatter(
            means_mu[idx],
            means_cov[idx],
            color=color,
            marker="o",
            s=size_ess[idx] if np.isfinite(size_ess[idx]) else 120.0,
            alpha=0.95,
            label=_label(m),
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )
        axes[1].errorbar(
            means_mu[idx],
            means_cov[idx],
            xerr=stds_mu[idx],
            yerr=stds_cov[idx],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=4,
            alpha=0.9,
        )
        axes[1].scatter(
            means_mu[idx],
            means_cov[idx],
            color=color,
            marker="o",
            s=size_wall[idx] if np.isfinite(size_wall[idx]) else 120.0,
            alpha=0.95,
            label=_label(m),
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
        )

    axes[0].set_title("Final: mean ± std (size = ESS)")
    axes[1].set_title("Final: mean ± std (size = wall time)")
    if np.all(np.array(means_mu) > 0) and np.all(np.array(means_cov) > 0):
        for ax in axes:
            ax.set_xscale("log")
            ax.set_yscale("log")
    for ax in axes:
        ax.set_xlabel(r"$\Vert\hat\mu\Vert_2$")
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
    fp = os.path.join(FIG_D, fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)

_plot_mu_cov_final_error_panels("pareto_mu_cov_final_mean_err.png")

def _plot_summary_panels(fname):
    latest = _latest_per_run(df_raw)
    if latest.empty:
        return
    metrics = [
        ("mu_err", r"$\Vert\hat\mu\Vert_2$"),
        ("cov_err", r"$\Vert\hat\Sigma-\Sigma\Vert_F$"),
        ("ksd", "KSD"),
        ("ess", "ESS"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))
    axes = axes.flatten()
    methods = [m for m in plot_methods if m in latest["method"].unique()]
    x = np.arange(len(methods))

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
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
        for j, m in enumerate(methods):
            ax.scatter(x[j], means[j], color=COLOR_MAP.get(m), s=60, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in methods], rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        if metric in {"mu_err", "cov_err", "ksd"} and np.all(np.array(means) > 0):
            ax.set_yscale("log")

    plt.tight_layout()
    fp = os.path.join(FIG_D, fname)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"saved → {fp}")
    plt.close(fig)

_plot_summary_panels("summary_metric_panels.png")

# 4) Final ESS comparison (use the best checkpoint per method/seed)
latest = _latest_per_run(df_raw)
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
ax.set_xticklabels([_label(m) for m in plot_methods], rotation=30, ha="right")
ax.set_xlabel("")
plt.ylabel("ESS (chain dim-0)")
plt.title("Final ESS (after last checkpoint)")
plt.tight_layout()
fp = os.path.join(FIG_D, "ess_bar.png")
plt.savefig(fp, dpi=150)
print(f"saved → {fp}")
plt.close()

# 5) ESS per gradient eval (final checkpoint)
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
ax.set_xticklabels([_label(m) for m in plot_methods], rotation=30, ha="right")
ax.set_xlabel("")
plt.ylabel("ESS / grad eval")
plt.title("Final ESS per gradient eval")
plt.tight_layout()
fp = os.path.join(FIG_D, "ess_per_grad_bar.png")
plt.savefig(fp, dpi=150)
print(f"saved → {fp}")
plt.close()
