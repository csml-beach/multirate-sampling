import argparse
import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))

from metrics_mixture2d import ess_1d, ksd_rbf, mean_log_prob, mode_stats
from samplers import (
    make_adaptive_multirate_svgd_step,
    make_multirate_svgd_step,
    make_sghmc_step,
    make_sgld_step,
    make_strang_svgd_step,
    make_svgd_step,
)
from targets_mixture2d import get_target, list_targets


DEFAULT_N_PARTICLES = 128
DEFAULT_N_ITER = 1_000
DEFAULT_SAVE_EVERY = 20
CHAIN_WINDOW = 100
DEFAULT_SEEDS = tuple(range(5))

LR_SVGD = 1e-2
LR_SGLD = 1e-2
LR_SGHMC = 1e-2
DEFAULT_PARTICLE_KERNEL = "rbf_multiscale"
DEFAULT_BW_SCALE = 0.5
DEFAULT_RBF_SCALES = (0.5, 1.0, 2.0)
ERR_TOL = 1e-2
INIT_CENTER = jnp.array([0.0, 0.0], dtype=jnp.float32)
INIT_STD = 0.5

DEFAULT_TARGETS = ("mix8",)
ALL_METHODS = (
    "multirate_svgd",
    "adaptive_multirate_svgd",
    "vanilla_svgd",
    "strang_svgd",
    "sgld",
    "sghmc",
)

OUT_DIR = Path("metrics") / "mixture2d"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GRID_L1_SIZE = 200


def _build_grid_reference(logp_fn, bounds, grid_size):
    xmin, xmax = bounds
    edges = np.linspace(xmin, xmax, grid_size + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centers, centers, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts)))
    logp = logp - np.max(logp)
    w = np.exp(logp)
    w_sum = w.sum()
    if w_sum == 0.0:
        raise ValueError("Grid reference density has zero mass.")
    ref = (w / w_sum).reshape(grid_size, grid_size)
    return edges, ref


def _grid_l1(samples, edges, ref):
    samples_np = np.asarray(samples)
    hist, _, _ = np.histogram2d(samples_np[:, 0], samples_np[:, 1], bins=[edges, edges])
    total = hist.sum()
    if total == 0.0:
        return float("nan")
    q = hist / total
    return float(np.sum(np.abs(q - ref)))


def _out_stem(target_name, particle_kernel, out_tag):
    kernel_suffix = {
        "rbf": "rbf",
        "imq": "imq",
        "rbf_multiscale": None,
    }[particle_kernel]
    suffix_parts = []
    if kernel_suffix is not None:
        suffix_parts.append(kernel_suffix)
    if out_tag:
        suffix_parts.append(out_tag)
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
    return f"{target_name}{suffix}"


def _build_samplers(logp, init_particles, x0_chain, *, particle_kernel, bw_scale, rbf_scales, imq_beta, imq_c):
    samplers = {
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
        ),
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
        ),
    }

    sgld_init_fn, sgld_step_fn = make_sgld_step(logp, LR_SGLD)
    samplers["sgld"] = (sgld_init_fn(x0_chain), sgld_step_fn)

    sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, LR_SGHMC)
    samplers["sghmc"] = (sghmc_init_fn(x0_chain), sghmc_step_fn)
    return samplers


def run_target(
    target_name,
    *,
    seeds,
    methods,
    n_particles,
    n_iter,
    save_every,
    particle_kernel,
    bw_scale,
    rbf_scales,
    imq_beta,
    imq_c,
    out_tag,
):
    logp, centers, bounds = get_target(target_name)
    grid_edges, grid_ref = _build_grid_reference(logp, bounds, GRID_L1_SIZE)
    score_fn = lambda x: jax.grad(lambda y: jnp.sum(logp(y)))(x)
    out_csv = OUT_DIR / f"{_out_stem(target_name, particle_kernel, out_tag)}.csv"

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target",
                "particle_kernel",
                "seed",
                "method",
                "iter",
                "grad_evals",
                "kernel_evals",
                "wall_s",
                "coverage",
                "entropy",
                "imbalance",
                "min_mass",
                "ess",
                "ksd",
                "mean_logp",
                "grid_l1",
                "nonfinite_frac",
                "m_used",
                "is_best",
            ]
        )

        for seed in seeds:
            seed_key = jax.random.PRNGKey(seed)
            seed_key, init_key, chain_key = jax.random.split(seed_key, 3)

            init_particles = INIT_CENTER + INIT_STD * jax.random.normal(init_key, (n_particles, 2))
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
            samplers = {name: spec for name, spec in samplers.items() if name in methods}

            chain_buffers = {}
            grad_eval_counts = {}
            kernel_eval_counts = {}

            for name, (state, step_fn) in samplers.items():
                print(f"\n▶ {target_name}:{particle_kernel} | seed {seed} | {name}")
                rng = seed_key
                t0 = time.time()
                grad_eval_counts[name] = 0.0
                kernel_eval_counts[name] = 0.0

                if name in {"sgld", "sghmc"}:
                    chain_buffers[name] = deque(maxlen=CHAIN_WINDOW)

                for it in trange(1, n_iter + 1):
                    rng, sub = jax.random.split(rng)
                    state, info = step_fn(state, sub)
                    grad_eval_counts[name] += float(info.get("grad_evals", 1.0))
                    kernel_eval_counts[name] += float(info.get("kernel_evals", 0.0))

                    if name in {"sgld", "sghmc"}:
                        chain_buffers[name].append(state[0])

                    if it % save_every == 0:
                        elapsed = time.time() - t0
                        if name in {"sgld", "sghmc"}:
                            samples = jnp.stack(list(chain_buffers[name]))
                        else:
                            samples = state if isinstance(state, jnp.ndarray) else state[0]

                        coverage, entropy, imbalance, min_mass = mode_stats(samples, centers)
                        ess_val = ess_1d(samples[:, 0])
                        ksd_val = ksd_rbf(samples, score_fn) if samples.shape[0] >= 2 else float("nan")
                        mlp_val = mean_log_prob(samples, logp)
                        grid_l1_val = _grid_l1(samples, grid_edges, grid_ref)
                        nonfinite_frac = float(info.get("nonfinite_frac", 0.0))
                        m_used = float(info.get("m_used", 0.0))

                        writer.writerow(
                            [
                                target_name,
                                particle_kernel,
                                seed,
                                name,
                                it,
                                grad_eval_counts[name],
                                kernel_eval_counts[name],
                                elapsed,
                                coverage,
                                entropy,
                                imbalance,
                                min_mass,
                                ess_val,
                                ksd_val,
                                mlp_val,
                                grid_l1_val,
                                nonfinite_frac,
                                m_used,
                                0,
                            ]
                        )
                        f.flush()
    print(f"saved -> {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed-budget multimodal Mix8 benchmark.")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS), choices=list_targets())
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=ALL_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES)
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument(
        "--particle-kernel",
        choices=["rbf", "imq", "rbf_multiscale"],
        default=DEFAULT_PARTICLE_KERNEL,
    )
    parser.add_argument("--bw-scale", type=float, default=DEFAULT_BW_SCALE)
    parser.add_argument("--rbf-scales", nargs="+", type=float, default=list(DEFAULT_RBF_SCALES))
    parser.add_argument("--imq-beta", type=float, default=0.5)
    parser.add_argument("--imq-c", type=float, default=1.0)
    parser.add_argument(
        "--out-tag",
        default=None,
        help="Optional suffix appended to output stems. Non-default kernels get auto-suffixed stems (for example, target_imq or target_msrbf).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for target_name in args.targets:
        run_target(
            target_name,
            seeds=args.seeds,
            methods=args.methods,
            n_particles=args.n_particles,
            n_iter=args.n_iter,
            save_every=args.save_every,
            particle_kernel=args.particle_kernel,
            bw_scale=args.bw_scale,
            rbf_scales=tuple(args.rbf_scales),
            imq_beta=args.imq_beta,
            imq_c=args.imq_c,
            out_tag=args.out_tag,
        )


if __name__ == "__main__":
    main()
