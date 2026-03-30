import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
    }
)

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))

from metrics_mixture2d import ksd_rbf, mean_log_prob, mode_stats
from samplers import (
    make_adaptive_multirate_svgd_step,
    make_multirate_svgd_step,
    make_sghmc_step,
    make_sgld_step,
    make_strang_svgd_step,
    make_svgd_step,
)
from targets_mixture2d import get_target


N_PARTICLES = 128
N_ITER = 1_000
CHAIN_WINDOW = 100
LR_SVGD = 1e-2
LR_SGLD = 1e-2
LR_SGHMC = 1e-2
DEFAULT_PARTICLE_KERNEL = "rbf_multiscale"
DEFAULT_BW_SCALE = 0.5
DEFAULT_RBF_SCALES = (0.5, 1.0, 2.0)
ERR_TOL = 1e-2
INIT_CENTER = jnp.array([0.0, 0.0], dtype=jnp.float32)
INIT_STD = 0.5

METHODS = [
    "adaptive_multirate_svgd",
    "multirate_svgd",
    "vanilla_svgd",
    "strang_svgd",
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

METHOD_COLORS = {
    "adaptive_multirate_svgd": "#d62728",
    "multirate_svgd": "#2ca02c",
    "vanilla_svgd": "#1f77b4",
    "strang_svgd": "#ff7f0e",
    "sgld": "#9467bd",
    "sghmc": "#8c564b",
}


def _format_ksd(value):
    if not np.isfinite(value):
        return "nan"
    if value >= 1e3 or value < 1e-2:
        return f"{value:.2e}"
    return f"{value:.3f}"


def _make_background(ax, logp_fn, bounds, grid=240):
    xmin, xmax = bounds
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(xmin, xmax, grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts))).reshape(grid, grid)
    levels = np.linspace(logp.max() - 10.0, logp.max(), 10)
    ax.contour(xx, yy, logp, levels=levels, cmap="viridis", linewidths=0.85)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")


def _build_samplers(logp, init_particles, x0_chain, *, particle_kernel, bw_scale, rbf_scales, imq_beta, imq_c):
    samplers = {
        "adaptive_multirate_svgd": (
            init_particles,
            make_adaptive_multirate_svgd_step(
                logp,
                base_dt=LR_SVGD,
                m_min=1,
                m_max=16,
                err_tol=ERR_TOL,
                bw_scale=bw_scale,
                kernel=particle_kernel,
                rbf_scales=rbf_scales,
                imq_beta=imq_beta,
                imq_c=imq_c,
            ),
            False,
        ),
        "multirate_svgd": (
            init_particles,
            make_multirate_svgd_step(
                logp,
                base_dt=LR_SVGD,
                m=4,
                bw_scale=bw_scale,
                kernel=particle_kernel,
                rbf_scales=rbf_scales,
                imq_beta=imq_beta,
                imq_c=imq_c,
            ),
            False,
        ),
        "vanilla_svgd": (
            init_particles,
            make_svgd_step(
                logp,
                LR_SVGD,
                bw_scale=bw_scale,
                kernel=particle_kernel,
                rbf_scales=rbf_scales,
                imq_beta=imq_beta,
                imq_c=imq_c,
            ),
            False,
        ),
        "strang_svgd": (
            init_particles,
            make_strang_svgd_step(
                logp,
                LR_SVGD,
                bw_scale=bw_scale,
                kernel=particle_kernel,
                rbf_scales=rbf_scales,
                imq_beta=imq_beta,
                imq_c=imq_c,
            ),
            False,
        ),
    }

    sgld_init_fn, sgld_step_fn = make_sgld_step(logp, LR_SGLD)
    samplers["sgld"] = (sgld_init_fn(x0_chain), sgld_step_fn, True)

    sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, LR_SGHMC)
    samplers["sghmc"] = (sghmc_init_fn(x0_chain), sghmc_step_fn, True)
    return samplers


def _run_method(name, state, step_fn, is_chain, logp, centers, score_fn, seed):
    rng = jax.random.PRNGKey(seed)
    chain_buffer = deque(maxlen=CHAIN_WINDOW) if is_chain else None

    for _ in range(N_ITER):
        rng, sub = jax.random.split(rng)
        state, _info = step_fn(state, sub)
        if is_chain:
            chain_buffer.append(np.asarray(state[0]))

    if is_chain:
        samples = jnp.asarray(np.stack(list(chain_buffer)))
    else:
        samples = state if isinstance(state, jnp.ndarray) else state[0]

    coverage, entropy, imbalance, min_mass = mode_stats(samples, centers)
    ksd = ksd_rbf(samples, score_fn) if samples.shape[0] >= 2 else float("nan")
    mean_logp = mean_log_prob(samples, logp)

    return {
        "method": name,
        "samples": np.asarray(samples),
        "coverage": float(coverage),
        "entropy": float(entropy),
        "imbalance": float(imbalance),
        "min_mass": float(min_mass),
        "ksd": float(ksd),
        "mean_logp": float(mean_logp),
        "is_chain": is_chain,
    }


def _write_summary_csv(rows, out_csv, particle_kernel):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["particle_kernel", "method", "coverage", "entropy", "imbalance", "min_mass", "ksd", "mean_logp", "n_points"]
        )
        for row in rows:
            writer.writerow(
                [
                    particle_kernel,
                    row["method"],
                    row["coverage"],
                    row["entropy"],
                    row["imbalance"],
                    row["min_mass"],
                    row["ksd"],
                    row["mean_logp"],
                    row["samples"].shape[0],
                ]
            )


def _default_outputs(seed, particle_kernel):
    kernel_suffix = {
        "rbf": "_rbf",
        "imq": "_imq",
        "rbf_multiscale": "",
    }[particle_kernel]
    stem = f"mix8{kernel_suffix}"
    base_dir = Path("figures") / "mixture2d" / stem
    return (
        base_dir / f"{stem}_final_particles_grid_seed{seed}.png",
        base_dir / f"{stem}_final_particles_grid_seed{seed}.csv",
    )


def make_mix8_grid(seed, out_png, out_csv, *, particle_kernel, bw_scale, rbf_scales, imq_beta, imq_c):
    logp, centers, bounds = get_target("mix8")
    score_fn = lambda x: jax.grad(lambda y: jnp.sum(logp(y)))(x)

    seed_key = jax.random.PRNGKey(seed)
    seed_key, init_key, chain_key = jax.random.split(seed_key, 3)
    init_particles = INIT_CENTER + INIT_STD * jax.random.normal(init_key, (N_PARTICLES, 2))
    x0_chain = INIT_CENTER + INIT_STD * jax.random.normal(chain_key, (2,))

    samplers = _build_samplers(
        logp,
        init_particles,
        x0_chain,
        particle_kernel=particle_kernel,
        bw_scale=bw_scale,
        rbf_scales=rbf_scales,
        imq_beta=imq_beta,
        imq_c=imq_c,
    )
    results = []
    for method in METHODS:
        state, step_fn, is_chain = samplers[method]
        results.append(_run_method(method, state, step_fn, is_chain, logp, centers, score_fn, seed))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.ravel()

    for ax, result in zip(axes, results):
        _make_background(ax, logp, bounds)
        ax.scatter(
            np.asarray(centers)[:, 0],
            np.asarray(centers)[:, 1],
            marker="x",
            s=55,
            c="black",
            linewidths=1.1,
            zorder=3,
        )
        ax.scatter(
            result["samples"][:, 0],
            result["samples"][:, 1],
            s=18 if not result["is_chain"] else 24,
            c=METHOD_COLORS[result["method"]],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.9,
            zorder=4,
        )
        ax.set_title(METHOD_LABELS[result["method"]], fontsize=16)
        ax.text(
            0.04,
            0.96,
            "\n".join(
                [
                    f"cov {result['coverage']:.3f}",
                    f"ent {result['entropy']:.3f}",
                    f"ksd {_format_ksd(result['ksd'])}",
                ]
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8", "boxstyle": "round,pad=0.25"},
        )

    for i, ax in enumerate(axes):
        if i % 3 == 0:
            ax.set_ylabel(r"$x_2$")
        if i >= 3:
            ax.set_xlabel(r"$x_1$")

    kernel_label = {"rbf": "RBF", "imq": "IMQ", "rbf_multiscale": "Multi-scale RBF"}[particle_kernel]
    fig.suptitle(rf"Mix8 final-budget comparison ({kernel_label} kernel, seed {seed}, 1000 steps)", fontsize=18)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    _write_summary_csv(results, out_csv, particle_kernel)
    print(f"saved -> {out_png}")
    print(f"saved -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Create a final-budget all-method Mix8 comparison grid.")
    parser.add_argument("--seed", type=int, default=0, help="Shared initialization seed.")
    parser.add_argument(
        "--particle-kernel",
        choices=["rbf", "imq", "rbf_multiscale"],
        default=DEFAULT_PARTICLE_KERNEL,
    )
    parser.add_argument("--bw-scale", type=float, default=DEFAULT_BW_SCALE)
    parser.add_argument("--rbf-scales", nargs="+", type=float, default=list(DEFAULT_RBF_SCALES))
    parser.add_argument("--imq-beta", type=float, default=0.5)
    parser.add_argument("--imq-c", type=float, default=1.0)
    parser.add_argument("--out", default=None, help="Output image path.")
    parser.add_argument("--summary-csv", default=None, help="Output CSV path for final metrics.")
    args = parser.parse_args()

    default_png, default_csv = _default_outputs(args.seed, args.particle_kernel)
    out_png = Path(args.out) if args.out else default_png
    out_csv = Path(args.summary_csv) if args.summary_csv else default_csv
    make_mix8_grid(
        args.seed,
        out_png,
        out_csv,
        particle_kernel=args.particle_kernel,
        bw_scale=args.bw_scale,
        rbf_scales=tuple(args.rbf_scales),
        imq_beta=args.imq_beta,
        imq_c=args.imq_c,
    )


if __name__ == "__main__":
    main()
