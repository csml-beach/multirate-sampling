# animate_mixture2d.py -----------------------------------------------------
import argparse
from pathlib import Path
from collections import deque
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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

from targets_mixture2d import get_target, list_targets
from samplers import (
    make_svgd_step,
    make_strang_svgd_step,
    make_sgld_step,
    make_sghmc_step,
    make_multirate_svgd_step,
    make_adaptive_multirate_svgd_step,
)
from metrics_mixture2d import ksd_rbf


def _build_sampler(name, logp, n_particles, lr_svgd, lr_sgld, lr_sghmc, err_tol, m_max, bw_scale, bounds, key):
    minv, maxv = bounds
    if name == "vanilla_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=minv, maxval=maxv)
        return state, make_svgd_step(logp, lr_svgd, bw_scale=bw_scale), False
    if name == "strang_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=minv, maxval=maxv)
        return state, make_strang_svgd_step(logp, lr_svgd, bw_scale=bw_scale), False
    if name == "multirate_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=minv, maxval=maxv)
        return state, make_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m=4,
            bw_scale=bw_scale,
        ), False
    if name == "adaptive_multirate_svgd":
        state = jax.random.uniform(key, (n_particles, 2), minval=minv, maxval=maxv)
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
        x0 = jax.random.uniform(key, (2,), minval=minv, maxval=maxv)
        return init_fn(x0), step_fn, True
    if name == "sghmc":
        init_fn, step_fn = make_sghmc_step(logp, lr_sghmc)
        x0 = jax.random.uniform(key, (2,), minval=minv, maxval=maxv)
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


def _make_background(ax, logp_fn, bounds, grid=200, cmap="mako", facecolor="#ffffff"):
    xmin, xmax = bounds
    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(xmin, xmax, grid)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts))).reshape(grid, grid)
    levels = np.linspace(logp.max() - 10, logp.max(), 10)
    cmap = _resolve_cmap(cmap)
    ax.set_facecolor(facecolor)
    ax.contourf(xx, yy, logp, levels=levels, cmap=cmap, alpha=0.95)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=20)


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
    lr_sgld,
    lr_sghmc,
    err_tol,
    m_max,
    bw_scale,
    early_stop,
    early_stop_tol,
    early_stop_patience,
    early_stop_min_checks,
    chain_window,
):
    logp, _centers, bounds = get_target(target)
    score_fn = lambda x: jax.grad(lambda y: jnp.sum(logp(y)))(x)
    key = jax.random.PRNGKey(seed)
    key, init_key = jax.random.split(key)
    state, step_fn, is_chain = _build_sampler(
        sampler,
        logp,
        n_particles,
        lr_svgd,
        lr_sgld,
        lr_sghmc,
        err_tol,
        m_max,
        bw_scale,
        bounds,
        init_key,
    )

    frames = []
    n_frames = max(1, n_steps // frame_every)
    best_ksd = None
    bad_checks = 0
    check_count = 0
    chain_buffer = deque(maxlen=chain_window) if is_chain else None

    total_steps = 0
    for _ in range(n_frames):
        for _ in range(frame_every):
            if total_steps >= n_steps:
                break
            key, sub = jax.random.split(key)
            state, _info = step_fn(state, sub)
            total_steps += 1
        if is_chain:
            chain_buffer.append(np.asarray(state[0]))
            frames.append(np.asarray(state[0]))
            samples = jnp.asarray(np.stack(list(chain_buffer))) if chain_buffer else None
        else:
            frames.append(np.asarray(state))
            samples = state if isinstance(state, jnp.ndarray) else state[0]

        if early_stop and samples is not None:
            check_count += 1
            ksd_val = ksd_rbf(samples, score_fn)
            if not np.isfinite(ksd_val):
                bad_checks += 1
            elif best_ksd is None or ksd_val < best_ksd:
                best_ksd = ksd_val
                bad_checks = 0
            elif check_count >= early_stop_min_checks and ksd_val > best_ksd * (1.0 + early_stop_tol):
                bad_checks += 1
            else:
                bad_checks = 0
            if check_count >= early_stop_min_checks and bad_checks >= early_stop_patience:
                best_str = f"{best_ksd:.3g}" if best_ksd is not None else "nan"
                print(
                    f"early stop: {target} | {sampler} at step {total_steps} "
                    f"(ksd {ksd_val:.3g} > best {best_str})"
                )
                break

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#ffffff")
    _make_background(ax, logp, bounds, grid=grid, cmap=cmap, facecolor="#ffffff")

    if is_chain:
        scatter = ax.scatter([], [], s=70, color="red", edgecolor="k", linewidth=0.4)
    else:
        scatter = ax.scatter([], [], s=26, color="red", edgecolor="k", linewidth=0.3, alpha=0.9)

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
    parser = argparse.ArgumentParser(description="Animate mixture2d sampler trajectories.")
    parser.add_argument("--target", default="all", help="Target name (mix8) or 'all'.")
    parser.add_argument(
        "--sampler",
        default="adaptive_multirate_svgd",
        help="Sampler name (vanilla_svgd/strang_svgd/multirate_svgd/adaptive_multirate_svgd/sgld/sghmc).",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Generate GIFs for all samplers for each target.",
    )
    parser.add_argument("--cmap", default="plasma_r", help="Matplotlib colormap for contours.")
    parser.add_argument("--n-steps", type=int, default=1000, help="Total sampler steps.")
    parser.add_argument("--frame-every", type=int, default=20, help="Steps between frames.")
    parser.add_argument("--n-particles", type=int, default=128, help="Particles for SVGD methods.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second.")
    parser.add_argument("--grid", type=int, default=200, help="Contour grid resolution.")
    parser.add_argument("--lr-svgd", type=float, default=1e-2, help="Step size for SVGD methods.")
    parser.add_argument("--lr-sgld", type=float, default=1e-2, help="Step size for SGLD.")
    parser.add_argument("--lr-sghmc", type=float, default=1e-2, help="Step size for SGHMC.")
    parser.add_argument("--err-tol", type=float, default=1e-2, help="Error tolerance for adaptive multirate.")
    parser.add_argument("--m-max", type=int, default=16, help="Max drift substeps for adaptive multirate.")
    parser.add_argument("--bw-scale", type=float, default=0.1, help="Kernel bandwidth scale (<1 strengthens repulsion).")
    parser.add_argument("--early-stop", action="store_true", help="Enable KSD-based early stopping.")
    parser.add_argument("--early-stop-tol", type=float, default=0.1, help="Relative KSD degradation tolerance.")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Bad checkpoints before stopping.")
    parser.add_argument("--early-stop-min-checks", type=int, default=5, help="Minimum checks before stopping.")
    parser.add_argument("--chain-window", type=int, default=100, help="Window for KSD on chain methods.")
    parser.add_argument(
        "--out",
        default="animations/mixture2d/anim.gif",
        help="Output GIF path.",
    )
    args = parser.parse_args()
    print(
        "matplotlib rcParams:",
        f"text.usetex={plt.rcParams.get('text.usetex')},",
        f"font.size={plt.rcParams.get('font.size')},",
        f"font.serif={plt.rcParams.get('font.serif')}",
    )

    targets = list_targets() if args.target == "all" else [args.target]
    samplers = [args.sampler]
    if args.all_methods:
        samplers = [
            "vanilla_svgd",
            "strang_svgd",
            "multirate_svgd",
            "adaptive_multirate_svgd",
            "sgld",
            "sghmc",
        ]

    for target in targets:
        for sampler in samplers:
            out_path = args.out
            if args.target == "all" or args.all_methods:
                out_path = f"animations/mixture2d/{target}/{sampler}.gif"
            run_animation(
                target=target,
                sampler=sampler,
                n_steps=args.n_steps,
                frame_every=args.frame_every,
                n_particles=args.n_particles,
                seed=args.seed,
                out_path=out_path,
                fps=args.fps,
                grid=args.grid,
                cmap=args.cmap,
                lr_svgd=args.lr_svgd,
                lr_sgld=args.lr_sgld,
                lr_sghmc=args.lr_sghmc,
                err_tol=args.err_tol,
                m_max=args.m_max,
                bw_scale=args.bw_scale,
                early_stop=args.early_stop,
                early_stop_tol=args.early_stop_tol,
                early_stop_patience=args.early_stop_patience,
                early_stop_min_checks=args.early_stop_min_checks,
                chain_window=args.chain_window,
            )


if __name__ == "__main__":
    main()
