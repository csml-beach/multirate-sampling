# benchmark_mixture2d.py ---------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np
import time
import csv
import os
import math
import sys
from pathlib import Path
from collections import deque
from tqdm import trange

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
from metrics_mixture2d import mode_stats, ess_1d, ksd_rbf, mean_log_prob


# ------------- configuration ---------------------------------------------
N_particles = 128
n_iter = 1_000
save_every = 20
chain_window = 100
SEEDS = range(5)

lr_svgd = 1e-2
lr_sgld = 1e-2
lr_sghmc = 1e-2
BW_SCALE = 0.5
ERR_TOL = 1e-2
EARLY_STOP = True
EARLY_STOP_TOL = 0.1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_CHECKS = 5

RUN_TARGETS = ["mix8"]
RUN_METHODS = None

OUT_DIR = os.path.join("metrics", "mixture2d")
os.makedirs(OUT_DIR, exist_ok=True)

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
    hist, _, _ = np.histogram2d(
        samples_np[:, 0],
        samples_np[:, 1],
        bins=[edges, edges],
    )
    total = hist.sum()
    if total == 0.0:
        return float("nan")
    q = hist / total
    return float(np.sum(np.abs(q - ref)))


def run_target(target_name, key):
    logp, centers, bounds = get_target(target_name)
    grid_edges, grid_ref = _build_grid_reference(logp, bounds, GRID_L1_SIZE)
    score_fn = lambda x: jax.grad(lambda y: jnp.sum(logp(y)))(x)

    for seed in SEEDS:
        seed_key = jax.random.PRNGKey(seed)
        seed_key, init_key, chain_key = jax.random.split(seed_key, 3)

        init_particles = jax.random.uniform(init_key, (N_particles, 2), minval=bounds[0], maxval=bounds[1])
        x0_chain = jax.random.uniform(chain_key, (2,), minval=bounds[0], maxval=bounds[1])

        samplers = {}
        samplers["multirate_svgd"] = (
            init_particles,
            make_multirate_svgd_step(
                logp,
                base_dt=lr_svgd,
                m=4,
                bw_scale=BW_SCALE,
            ),
        )

        samplers["adaptive_multirate_svgd"] = (
            init_particles,
            make_adaptive_multirate_svgd_step(
                logp,
                base_dt=lr_svgd,
                m_min=1,
                m_max=16,
                err_tol=ERR_TOL,
                bw_scale=BW_SCALE,
            ),
        )

        samplers["vanilla_svgd"] = (
            init_particles,
            make_svgd_step(logp, lr_svgd, bw_scale=BW_SCALE),
        )

        samplers["strang_svgd"] = (
            init_particles,
            make_strang_svgd_step(logp, lr_svgd, bw_scale=BW_SCALE),
        )

        sgld_init_fn, sgld_step_fn = make_sgld_step(logp, lr_sgld)
        samplers["sgld"] = (sgld_init_fn(x0_chain), sgld_step_fn)

        sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, lr_sghmc)
        samplers["sghmc"] = (sghmc_init_fn(x0_chain), sghmc_step_fn)

        if RUN_METHODS is not None:
            samplers = {k: v for k, v in samplers.items() if k in RUN_METHODS}

        out_csv = os.path.join(OUT_DIR, f"{target_name}.csv")
        with open(out_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(
                    [
                        "target",
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

            chain_buffers = {}
            grad_eval_counts = {}
            kernel_eval_counts = {}

            for name, (state, step_fn) in samplers.items():
                print(f"\n▶ {target_name} | seed {seed} | {name}")
                rng = seed_key
                t0 = time.time()
                grad_eval_counts[name] = 0.0
                kernel_eval_counts[name] = 0.0
                best_ksd = None
                best_row = None
                best_iter = None
                last_iter = None
                bad_checks = 0
                check_count = 0

                if name in {"sgld", "sghmc"}:
                    chain_buffers[name] = deque(maxlen=chain_window)

                for it in trange(1, n_iter + 1):
                    rng, sub = jax.random.split(rng)
                    state, info = step_fn(state, sub)
                    grad_eval_counts[name] += float(info.get("grad_evals", 1.0))
                    kernel_eval_counts[name] += float(info.get("kernel_evals", 0.0))

                    if it % save_every == 0:
                        elapsed = time.time() - t0
                        if name in {"sgld", "sghmc"}:
                            chain_buffers[name].append(state[0])
                            samples = jnp.stack(list(chain_buffers[name]))
                        else:
                            samples = state if isinstance(state, jnp.ndarray) else state[0]

                        coverage, entropy, imbalance, min_mass = mode_stats(samples, centers)
                        ess_val = ess_1d(samples[:, 0])
                        ksd_val = ksd_rbf(samples, score_fn)
                        mlp_val = mean_log_prob(samples, logp)
                        grid_l1_val = _grid_l1(samples, grid_edges, grid_ref)
                        nonfinite_frac = float(info.get("nonfinite_frac", 0.0))
                        m_used = float(info.get("m_used", 0.0))

                        writer.writerow(
                            [
                                target_name,
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
                        last_iter = it
                        if EARLY_STOP:
                            check_count += 1
                            if not math.isfinite(ksd_val):
                                bad_checks += 1
                            elif best_ksd is None or ksd_val < best_ksd:
                                best_ksd = ksd_val
                                best_row = [
                                    target_name,
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
                                    1,
                                ]
                                best_iter = it
                                bad_checks = 0
                            elif check_count >= EARLY_STOP_MIN_CHECKS and ksd_val > best_ksd * (1.0 + EARLY_STOP_TOL):
                                bad_checks += 1
                            else:
                                bad_checks = 0
                            if check_count >= EARLY_STOP_MIN_CHECKS and bad_checks >= EARLY_STOP_PATIENCE:
                                best_str = f"{best_ksd:.3g}" if best_ksd is not None else "nan"
                                print(
                                    f"early stop: {target_name} | {name} at iter {it} "
                                    f"(ksd {ksd_val:.3g} > best {best_str})"
                                )
                                break

                if best_row is not None and last_iter is not None and best_iter != last_iter:
                    writer.writerow(best_row)
                    f.flush()


def main():
    key = jax.random.PRNGKey(0)
    targets = RUN_TARGETS if RUN_TARGETS else list_targets()
    for _i, tname in enumerate(targets):
        key, sub = jax.random.split(key)
        run_target(tname, sub)


if __name__ == "__main__":
    main()
