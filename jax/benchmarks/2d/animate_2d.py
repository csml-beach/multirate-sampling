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


def _build_sampler(name, logp, n_particles, lr_svgd, lr_sgld, lr_sghmc, err_tol, m_max, bw_scale, key):
    if name == "vanilla_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=-4.0, maxval=4.0)
        return state, make_svgd_step(logp, lr_svgd, bw_scale=bw_scale), False
    if name == "strang_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=-4.0, maxval=4.0)
        return state, make_strang_svgd_step(logp, lr_svgd, bw_scale=bw_scale), False
    if name == "multirate_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=-4.0, maxval=4.0)
        return state, make_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m=4,
            L_inv=None,
            bw_scale=bw_scale,
        ), False
    if name == "adaptive_multirate_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=-4.0, maxval=4.0)
        step = make_adaptive_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m_min=1,
            m_max=m_max,
            err_tol=err_tol,
            L_inv=None,
            bw_scale=bw_scale,
        )
        return state, step, False
    if name == "sgld":
        init_fn, step_fn = make_sgld_step(logp, lr_sgld)
        x0 = jax.random.uniform(key, (2,), minval=-4.0, maxval=4.0)
        return init_fn(x0), step_fn, True
    if name == "sghmc":
        init_fn, step_fn = make_sghmc_step(logp, lr_sghmc)
        x0 = jax.random.uniform(key, (2,), minval=-4.0, maxval=4.0)
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


def _make_background(ax, logp_fn, bounds, grid=200, cmap="mako"):
    xmin, xmax = bounds
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(xmin, xmax, grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts))).reshape(grid, grid)
    levels = np.linspace(logp.max() - 10, logp.max(), 24)
    cmap = _resolve_cmap(cmap)
    ax.contourf(xx, yy, logp, levels=levels, cmap=cmap, alpha=0.95)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
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
):
    logp, _score_fn, _mean_ref, _cov_ref, bounds = get_target(target)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state, step_fn, is_chain = _build_sampler(
        sampler, logp, n_particles, lr_svgd, 1e-4, 1e-4, err_tol, m_max, bw_scale, init_key
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
    _make_background(ax, logp, bounds, grid=grid, cmap=cmap)

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
        )


if __name__ == "__main__":
    main()
