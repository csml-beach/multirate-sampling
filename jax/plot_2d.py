# plot_2d.py ---------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


TARGET = "banana"
CSV = os.path.join("metrics_2d", f"{TARGET}.csv")
FIG_D = "figures_2d"
os.makedirs(FIG_D, exist_ok=True)

df = pd.read_csv(CSV)

plot_methods = [
    "vanilla_svgd",
    "strang_svgd",
    "multirate_svgd",
    "adaptive_multirate_svgd",
    "sgld",
    "sghmc",
]
df = df[df["method"].isin(plot_methods)]

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


def smooth_data(x, y, window=15):
    if len(y) < window:
        return x, y
    y_padded = np.concatenate([np.repeat(y[0], window // 2), y, np.repeat(y[-1], window // 2)])
    y_smooth = np.convolve(y_padded, np.ones(window) / window, mode="valid")
    return x, y_smooth


def _plot(metric, ylabel, fname, logy=True, smooth=True, window=15, use_kernel_evals=False):
    x_col, x_label = _resolve_xaxis(use_kernel_evals)
    plt.figure(figsize=(9, 4))
    for name, sub in df.groupby("method"):
        x, y = sub[x_col].values, sub[metric].values
        if smooth:
            x, y = smooth_data(x, y, window)
        plt.plot(
            x,
            y,
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
    plt.title(f"{ylabel} vs {x_label.lower()} ({TARGET}){title_suffix}")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.78, 1])
    fp = os.path.join(FIG_D, fname)
    plt.savefig(fp, dpi=150)
    print(f"saved → {fp}")


def _fname(base, use_kernel):
    return f"{TARGET}_{base}_kernel.png" if use_kernel else f"{TARGET}_{base}.png"


_plot("mu_err", r"$\Vert\hat\mu-\mu\Vert_2$", _fname("mu_error_curve", False))
_plot(
    "mu_err",
    r"$\Vert\hat\mu-\mu\Vert_2$",
    _fname("mu_error_curve", True),
    use_kernel_evals=True,
)

_plot("cov_err", r"$\Vert\hat\Sigma-\Sigma\Vert_F$", _fname("cov_error_curve", False))
_plot(
    "cov_err",
    r"$\Vert\hat\Sigma-\Sigma\Vert_F$",
    _fname("cov_error_curve", True),
    use_kernel_evals=True,
)

if "ksd" in df.columns:
    _plot("ksd", "KSD", _fname("ksd_curve", False), logy=True)
    _plot("ksd", "KSD", _fname("ksd_curve", True), logy=True, use_kernel_evals=True)

if "ess" in df.columns:
    _plot("ess", "ESS", _fname("ess_curve", False), logy=False)
    _plot("ess", "ESS", _fname("ess_curve", True), logy=False, use_kernel_evals=True)

latest = df.sort_values("iter").groupby("method").tail(1)
plt.figure(figsize=(9.5, 4.5))
sns.barplot(data=latest, x="method", y="ess", hue="method", palette="Set2", legend=False)
plt.ylabel("ESS (chain dim-0)")
plt.title(f"Final ESS ({TARGET})")
plt.tight_layout()
fp = os.path.join(FIG_D, f"{TARGET}_ess_bar.png")
plt.savefig(fp, dpi=150)
print(f"saved → {fp}")

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
plt.title(f"Final ESS per gradient eval ({TARGET})")
plt.tight_layout()
fp = os.path.join(FIG_D, f"{TARGET}_ess_per_grad_bar.png")
plt.savefig(fp, dpi=150)
print(f"saved → {fp}")
