import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))

from targets_2d import get_target, list_targets
from samplers import make_adaptive_multirate_svgd_step


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 16,
    }
)


def make_demo_step(logp_fn, base_dt, bw_scale, err_tol, m_min, m_max):
    return make_adaptive_multirate_svgd_step(
        logp_fn,
        base_dt=base_dt,
        m_min=m_min,
        m_max=m_max,
        err_tol=err_tol,
        bw_scale=bw_scale,
    )


def _resolve_cmap(name):
    if name in plt.colormaps():
        return name
    try:
        import seaborn as sns
    except ImportError:
        return "magma"
    return sns.color_palette(name, as_cmap=True)


def _make_levels(logp, n_levels, level_span, level_focus):
    vmax = float(np.max(logp))
    vmin = vmax - float(level_span)
    u = np.linspace(0.0, 1.0, int(n_levels))
    # level_focus < 1.0 concentrates contours near high-density regions.
    levels = vmin + (u ** float(level_focus)) * (vmax - vmin)
    return levels


def _split_bounds(bounds):
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and np.isscalar(bounds[0])
        and np.isscalar(bounds[1])
    ):
        xmin, xmax = float(bounds[0]), float(bounds[1])
        ymin, ymax = xmin, xmax
        return xmin, xmax, ymin, ymax
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and isinstance(bounds[0], tuple)
        and isinstance(bounds[1], tuple)
        and len(bounds[0]) == 2
        and len(bounds[1]) == 2
    ):
        xmin, xmax = float(bounds[0][0]), float(bounds[0][1])
        ymin, ymax = float(bounds[1][0]), float(bounds[1][1])
        return xmin, xmax, ymin, ymax
    raise ValueError(f"Invalid bounds format: {bounds!r}")


def _draw_density(ax, logp_fn, bounds, grid, cmap, n_levels, level_span, level_focus):
    xmin, xmax, ymin, ymax = _split_bounds(bounds)
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(ymin, ymax, grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts))).reshape(grid, grid)
    levels = _make_levels(logp, n_levels=n_levels, level_span=level_span, level_focus=level_focus)
    ax.contour(
        xx,
        yy,
        logp,
        levels=levels,
        cmap=_resolve_cmap(cmap),
        linewidths=0.5,
        alpha=0.9,
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")


def _resolve_plot_bounds(default_bounds, args):
    values = (args.plot_xmin, args.plot_xmax, args.plot_ymin, args.plot_ymax)
    if all(v is None for v in values):
        return default_bounds
    if any(v is None for v in values):
        raise ValueError("Specify all of --plot-xmin/--plot-xmax/--plot-ymin/--plot-ymax together.")
    return ((float(args.plot_xmin), float(args.plot_xmax)), (float(args.plot_ymin), float(args.plot_ymax)))


def _run_demo(logp_fn, init_particles, iters, seed, step_fn):
    key = jax.random.PRNGKey(seed)
    state = init_particles
    for _ in range(iters):
        key, sub = jax.random.split(key)
        state, _ = step_fn(state, sub)
    return state


def _plot_target(target, args):
    logp, _score, _mean, _cov, target_bounds = get_target(target)
    plot_bounds = _resolve_plot_bounds(target_bounds, args)
    init_bounds = plot_bounds if args.init_bounds == "plot" else target_bounds
    xmin, xmax, ymin, ymax = _split_bounds(init_bounds)
    init_min = jnp.asarray([xmin, ymin], dtype=jnp.float32)
    init_max = jnp.asarray([xmax, ymax], dtype=jnp.float32)
    key = jax.random.PRNGKey(args.seed)
    init_particles = jax.random.uniform(
        key,
        (args.particles, 2),
        minval=init_min,
        maxval=init_max,
    )

    step_fn = make_demo_step(
        logp_fn=logp,
        base_dt=args.lr,
        bw_scale=args.bw_scale,
        err_tol=args.err_tol,
        m_min=args.m_min,
        m_max=args.m_max,
    )
    final_particles = _run_demo(
        logp_fn=logp,
        init_particles=init_particles,
        iters=args.iters,
        seed=args.seed,
        step_fn=step_fn,
    )

    init_np = np.asarray(init_particles)
    final_np = np.asarray(final_particles)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    # Panel A: density + initial particles
    _draw_density(
        axes[0],
        logp,
        plot_bounds,
        grid=args.grid,
        cmap=args.cmap,
        n_levels=args.n_levels,
        level_span=args.level_span,
        level_focus=args.level_focus,
    )
    axes[0].scatter(
        init_np[:, 0],
        init_np[:, 1],
        s=args.marker_size,
        color="red",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
    )
    axes[0].set_title("Initial Particles")

    # Panel B: density + short-run final
    _draw_density(
        axes[1],
        logp,
        plot_bounds,
        grid=args.grid,
        cmap=args.cmap,
        n_levels=args.n_levels,
        level_span=args.level_span,
        level_focus=args.level_focus,
    )
    axes[1].scatter(
        final_np[:, 0],
        final_np[:, 1],
        s=args.marker_size,
        color="red",
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
    )
    axes[1].set_title("Final Particles")

    for ax in axes:
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")

    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target}_demo.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2-panel demo figures (density + init, spread-preserving final) without writing metrics CSVs."
    )
    parser.add_argument("--targets", type=str, default="all", help="Comma-separated target list or 'all'.")
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--iters", type=int, default=140, help="Short-run iteration count for panel B.")
    parser.add_argument("--lr", type=float, default=7e-3)
    parser.add_argument("--bw-scale", type=float, default=0.1)
    parser.add_argument("--err-tol", type=float, default=1e-2)
    parser.add_argument("--m-min", type=int, default=1)
    parser.add_argument("--m-max", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=220)
    parser.add_argument("--n-levels", type=int, default=14, help="Number of contour levels.")
    parser.add_argument(
        "--level-span",
        type=float,
        default=10.0,
        help="Log-density span below max used for contours.",
    )
    parser.add_argument(
        "--level-focus",
        type=float,
        default=1.0,
        help="Contour spacing in log-density space; <1 shifts levels toward high density.",
    )
    parser.add_argument("--plot-xmin", type=float, default=None, help="Optional plotting x-min override.")
    parser.add_argument("--plot-xmax", type=float, default=None, help="Optional plotting x-max override.")
    parser.add_argument("--plot-ymin", type=float, default=None, help="Optional plotting y-min override.")
    parser.add_argument("--plot-ymax", type=float, default=None, help="Optional plotting y-max override.")
    parser.add_argument(
        "--init-bounds",
        type=str,
        choices=("target", "plot"),
        default="target",
        help="Initialization bounds source when plot bounds are overridden.",
    )
    parser.add_argument("--marker-size", type=float, default=17.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--cmap", type=str, default="mako")
    parser.add_argument("--out-dir", type=str, default="figures/2d_demo")
    args = parser.parse_args()
    if args.level_span <= 0.0:
        raise ValueError("--level-span must be > 0.")
    if args.level_focus <= 0.0:
        raise ValueError("--level-focus must be > 0.")

    if args.targets.strip().lower() == "all":
        targets = list_targets()
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    for t in targets:
        _plot_target(t, args)


if __name__ == "__main__":
    main()
