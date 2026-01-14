# plot_gauss50.py -----------------------------------------------------------
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os, textwrap
import numpy as np

CSV   = os.path.join("metrics_50d", "metrics_gauss50.csv")
FIG_D = "figures_50d"
os.makedirs(FIG_D, exist_ok=True)

df = pd.read_csv(CSV)

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
df = df[df["method"].isin(plot_methods)]

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

def _resolve_xaxis(use_kernel_evals):
    if use_kernel_evals and "kernel_evals" in df.columns:
        return "kernel_evals", "Kernel evaluations"
    if "grad_evals" in df.columns:
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
    x_col, x_label = _resolve_xaxis(use_kernel_evals)
    plt.figure(figsize=(9, 4))
    for name, sub in df.groupby("method"):
        x, y = sub[x_col].values, sub[metric].values
        if smooth:
            x, y = smooth_data(x, y, window)
        plt.plot(
            x, y,
            label=name,
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

def _plot_dual_x(metric, ylabel, fname, logy=True, smooth=True, window=7):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    x_col, x_label = _resolve_xaxis(False)
    for name, sub in df.groupby("method"):
        x, y = sub[x_col].values, sub[metric].values
        if smooth:
            x, y = smooth_data(x, y, window)
        axes[0].plot(
            x,
            y,
            label=name,
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
            x, y = sub["kernel_evals"].values, sub[metric].values
            if smooth:
                x, y = smooth_data(x, y, window)
            axes[1].plot(
                x,
                y,
                label=name,
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

# 1) mean-error curve (dual-axis)
_plot_dual_x("mu_err", r"$\Vert\hat\mu\Vert_2$", "mu_error_dual_axis.png")

# 2) covariance Frobenius error curve (dual-axis)
_plot_dual_x("cov_err", r"$\Vert\hat\Sigma-\Sigma\Vert_F$", "cov_error_dual_axis.png")

# 3) KSD curve (if available, dual-axis)
if "ksd" in df.columns:
    _plot_dual_x("ksd", "KSD", "ksd_dual_axis.png", logy=True)

# 4) Final ESS comparison (use the last recorded point of each method)
latest = df.sort_values("iter").groupby("method").tail(1)
plt.figure(figsize=(9.5, 4.5))
sns.barplot(data=latest, x="method", y="ess", hue="method", palette="Set2", legend=False)
plt.ylabel("ESS (chain dim-0)")
plt.title("Final ESS (after last checkpoint)")
plt.tight_layout()
fp = os.path.join(FIG_D, "ess_bar.png")
plt.savefig(fp, dpi=150)
print(f"saved → {fp}")

# 5) ESS per gradient eval (final checkpoint)
latest = latest.copy()
_grad_col = "grad_evals" if "grad_evals" in latest.columns else "iter"
latest["ess_per_grad"] = latest["ess"] / latest[_grad_col].replace(0, np.nan)
plt.figure(figsize=(9.5, 4.5))
sns.barplot(
    data=latest,
    x="method",
    y="ess_per_grad",
    hue="method",
    palette="Set2",
    legend=False,
)
plt.ylabel("ESS / grad eval")
plt.title("Final ESS per gradient eval")
plt.tight_layout()
fp = os.path.join(FIG_D, "ess_per_grad_bar.png")
plt.savefig(fp, dpi=150)
print(f"saved → {fp}")
