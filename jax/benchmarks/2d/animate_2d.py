# animate_2d.py ------------------------------------------------------------
import argparse
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation

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

from targets_2d import get_target, list_targets
from samplers import (
    make_svgd_step,
    make_strang_svgd_step,
    make_sgld_step,
    make_sghmc_step,
    make_multirate_svgd_step,
    make_adaptive_multirate_svgd_step,
)


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


def _build_sampler(
    name,
    logp,
    n_particles,
    lr_svgd,
    lr_sgld,
    lr_sghmc,
    err_tol,
    m_max,
    bw_scale,
    key,
    init_min,
    init_max,
):
    if name == "vanilla_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=init_min, maxval=init_max)
        return state, make_svgd_step(logp, lr_svgd, bw_scale=bw_scale), False
    if name == "strang_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=init_min, maxval=init_max)
        return state, make_strang_svgd_step(logp, lr_svgd, bw_scale=bw_scale), False
    if name == "multirate_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=init_min, maxval=init_max)
        return state, make_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m=4,
            bw_scale=bw_scale,
        ), False
    if name == "adaptive_multirate_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=init_min, maxval=init_max)
        step = make_adaptive_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m_min=1,
            m_max=m_max,
            err_tol=err_tol,
            bw_scale=bw_scale,
        )
        return state, step, False
    if name == "sgld":
        init_fn, step_fn = make_sgld_step(logp, lr_sgld)
        x0 = jax.random.uniform(key, (2,), minval=init_min, maxval=init_max)
        return init_fn(x0), step_fn, True
    if name == "sghmc":
        init_fn, step_fn = make_sghmc_step(logp, lr_sghmc)
        x0 = jax.random.uniform(key, (2,), minval=init_min, maxval=init_max)
        return init_fn(x0), step_fn, True
    raise ValueError(f"Unknown sampler '{name}'")


def _resolve_cmap(name):
    if name in plt.colormaps():
        return name
    try:
        import seaborn as sns
    except ImportError:
        print(f"note: cmap '{name}' not available, falling back to 'magma'")
        return "magma"
    return sns.color_palette(name, as_cmap=True)


def _make_levels(logp, n_levels, level_span, level_focus):
    vmax = float(np.max(logp))
    vmin = vmax - float(level_span)
    u = np.linspace(0.0, 1.0, int(n_levels))
    levels = vmin + (u ** float(level_focus)) * (vmax - vmin)
    return levels


def _resolve_plot_bounds(default_bounds, plot_xmin, plot_xmax, plot_ymin, plot_ymax):
    values = (plot_xmin, plot_xmax, plot_ymin, plot_ymax)
    if all(v is None for v in values):
        return default_bounds
    if any(v is None for v in values):
        raise ValueError("Specify all of --plot-xmin/--plot-xmax/--plot-ymin/--plot-ymax together.")
    return ((float(plot_xmin), float(plot_xmax)), (float(plot_ymin), float(plot_ymax)))


def _make_background(
    ax,
    logp_fn,
    bounds,
    grid=200,
    cmap="mako",
    contour_style="filled",
    n_levels=24,
    level_span=10.0,
    level_focus=1.0,
):
    xmin, xmax, ymin, ymax = _split_bounds(bounds)
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(ymin, ymax, grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts))).reshape(grid, grid)
    levels = _make_levels(logp, n_levels=n_levels, level_span=level_span, level_focus=level_focus)
    cmap = _resolve_cmap(cmap)
    if contour_style == "lines":
        ax.contour(xx, yy, logp, levels=levels, cmap=cmap, linewidths=0.7, alpha=0.95)
    else:
        ax.contourf(xx, yy, logp, levels=levels, cmap=cmap, alpha=0.95)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")


def run_animation(
    target,
    sampler,
    n_steps,
    frame_every,
    n_particles,
    seed,
    out_path,
    fps,
    grid,
    cmap,
    lr_svgd,
    err_tol,
    m_max,
    bw_scale,
    plot_xmin,
    plot_xmax,
    plot_ymin,
    plot_ymax,
    init_bounds,
    contour_style,
    n_levels,
    level_span,
    level_focus,
):
    logp, _score_fn, _mean_ref, _cov_ref, target_bounds = get_target(target)
    plot_bounds = _resolve_plot_bounds(
        target_bounds,
        plot_xmin=plot_xmin,
        plot_xmax=plot_xmax,
        plot_ymin=plot_ymin,
        plot_ymax=plot_ymax,
    )
    active_init_bounds = plot_bounds if init_bounds == "plot" else target_bounds
    xmin, xmax, ymin, ymax = _split_bounds(active_init_bounds)
    init_min = jnp.asarray([xmin, ymin], dtype=jnp.float32)
    init_max = jnp.asarray([xmax, ymax], dtype=jnp.float32)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state, step_fn, is_chain = _build_sampler(
        sampler,
        logp,
        n_particles,
        lr_svgd,
        1e-4,
        1e-4,
        err_tol,
        m_max,
        bw_scale,
        init_key,
        init_min,
        init_max,
    )

    frames = []
    n_frames = max(1, n_steps // frame_every)
    for _ in range(n_frames):
        for _ in range(frame_every):
            key, sub = jax.random.split(key)
            state, _info = step_fn(state, sub)
        if is_chain:
            frames.append(np.asarray(state[0]))
        else:
            frames.append(np.asarray(state))

    fig, ax = plt.subplots(figsize=(6, 6))
    _make_background(
        ax,
        logp,
        plot_bounds,
        grid=grid,
        cmap=cmap,
        contour_style=contour_style,
        n_levels=n_levels,
        level_span=level_span,
        level_focus=level_focus,
    )

    if is_chain:
        scatter = ax.scatter([], [], s=70, color="red", edgecolor="k", linewidth=0.4)
    else:
        scatter = ax.scatter([], [], s=22, color="red", edgecolor="k", linewidth=0.3, alpha=0.9)

    def update(i):
        data = frames[i]
        if is_chain:
            scatter.set_offsets(data[None, :])
        else:
            scatter.set_offsets(data)
        return (scatter,)

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    print(f"saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Animate 2D sampler trajectories.")
    parser.add_argument(
        "--target",
        default="all",
        help="Target name (banana/ring/squiggly/two_moons/funnel) or 'all'.",
    )
    parser.add_argument(
        "--sampler",
        default="adaptive_multirate_svgd",
        help="Sampler name (vanilla_svgd/strang_svgd/multirate_svgd/adaptive_multirate_svgd/sgld/sghmc).",
    )
    parser.add_argument("--cmap", default="mako", help="Matplotlib colormap for contours.")
    parser.add_argument("--n-steps", type=int, default=300, help="Total sampler steps.")
    parser.add_argument("--frame-every", type=int, default=10, help="Steps between frames.")
    parser.add_argument("--n-particles", type=int, default=128, help="Particles for SVGD methods.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second.")
    parser.add_argument("--grid", type=int, default=200, help="Contour grid resolution.")
    parser.add_argument("--contour-style", choices=("filled", "lines"), default="filled", help="Background contour style.")
    parser.add_argument("--n-levels", type=int, default=24, help="Number of contour levels.")
    parser.add_argument("--level-span", type=float, default=10.0, help="Log-density span below max used for contours.")
    parser.add_argument("--level-focus", type=float, default=1.0, help="Contour spacing in log-density space; <1 shifts levels toward high density.")
    parser.add_argument("--plot-xmin", type=float, default=None, help="Optional plotting x-min override.")
    parser.add_argument("--plot-xmax", type=float, default=None, help="Optional plotting x-max override.")
    parser.add_argument("--plot-ymin", type=float, default=None, help="Optional plotting y-min override.")
    parser.add_argument("--plot-ymax", type=float, default=None, help="Optional plotting y-max override.")
    parser.add_argument("--init-bounds", choices=("target", "plot"), default="target", help="Initialization bounds source when plot bounds are overridden.")
    parser.add_argument("--lr-svgd", type=float, default=2e-4, help="Step size for SVGD methods.")
    parser.add_argument("--err-tol", type=float, default=1e-2, help="Error tolerance for adaptive multirate.")
    parser.add_argument("--m-max", type=int, default=16, help="Max drift substeps for adaptive multirate.")
    parser.add_argument("--bw-scale", type=float, default=0.5, help="Kernel bandwidth scale (<1 strengthens repulsion).")
    parser.add_argument(
        "--out",
        default="animations/2d/anim.gif",
        help="Output GIF path.",
    )
    args = parser.parse_args()
    if args.level_span <= 0.0:
        raise ValueError("--level-span must be > 0.")
    if args.level_focus <= 0.0:
        raise ValueError("--level-focus must be > 0.")

    targets = list_targets() if args.target == "all" else [args.target]
    for target in targets:
        out_path = args.out
        if args.target == "all":
            out_path = f"animations/2d/{target}/{args.sampler}.gif"
        run_animation(
            target=target,
            sampler=args.sampler,
            n_steps=args.n_steps,
            frame_every=args.frame_every,
            n_particles=args.n_particles,
            seed=args.seed,
            out_path=out_path,
            fps=args.fps,
            grid=args.grid,
            cmap=args.cmap,
            lr_svgd=args.lr_svgd,
            err_tol=args.err_tol,
            m_max=args.m_max,
            bw_scale=args.bw_scale,
            plot_xmin=args.plot_xmin,
            plot_xmax=args.plot_xmax,
            plot_ymin=args.plot_ymin,
            plot_ymax=args.plot_ymax,
            init_bounds=args.init_bounds,
            contour_style=args.contour_style,
            n_levels=args.n_levels,
            level_span=args.level_span,
            level_focus=args.level_focus,
        )


if __name__ == "__main__":
    main()
