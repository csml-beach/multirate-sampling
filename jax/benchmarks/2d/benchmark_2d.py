# benchmark_2d.py ----------------------------------------------------------
import jax
import jax.numpy as jnp
import numpy as np
import time
import csv
import os
import math
import sys
from pathlib import Path

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))
from collections import deque
from tqdm import trange

from targets_2d import get_target, list_targets
from samplers import (
    make_svgd_step,
    make_strang_svgd_step,
    make_sgld_step,
    make_sghmc_step,
    make_multirate_svgd_step,
    make_adaptive_multirate_svgd_step,
)
from metrics_2d import cov_error, ess_1d, ksd_rbf, mean_log_prob


# ------------- configuration ---------------------------------------------
N_particles = 128
n_iter = 1_000
save_every = 20
chain_window = 100
SEEDS = range(5)

lr_svgd = 1e-2
lr_sgld = 1e-2
lr_sghmc = 1e-2
BW_SCALE = 0.1  # <1.0 strengthens repulsion (smaller bandwidth)
ERR_TOL = 1e-2
EARLY_STOP = True          # 2D is a pure sampling benchmark: monitor KSD
EARLY_STOP_TOL = 0.1      # Relative degradation allowed (5%)
EARLY_STOP_PATIENCE = 5    # Number of bad checkpoints before stopping
EARLY_STOP_MIN_CHECKS = 5  # Minimum checkpoints before applying early stop
EARLY_STOP_NONFINITE_PATIENCE = 2  # Stop after repeated non-finite KSD values

RUN_TARGETS = ["banana", "ring", "squiggly", "two_moons", "funnel"]
RUN_METHODS = None  # set to a list of method names to filter

OUT_DIR = os.path.join("metrics", "2d")
os.makedirs(OUT_DIR, exist_ok=True)

GRID_L1_SIZE = 200


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


def _build_grid_reference(logp_fn, bounds, grid_size):
    xmin, xmax, ymin, ymax = _split_bounds(bounds)
    x_edges = np.linspace(xmin, xmax, grid_size + 1)
    y_edges = np.linspace(ymin, ymax, grid_size + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logp = np.asarray(logp_fn(jnp.asarray(pts)))
    logp = logp - np.max(logp)
    w = np.exp(logp)
    w_sum = w.sum()
    if w_sum == 0.0:
        raise ValueError("Grid reference density has zero mass.")
    ref = (w / w_sum).reshape(grid_size, grid_size)
    return x_edges, y_edges, ref


def _grid_l1(samples, x_edges, y_edges, ref):
    samples_np = np.asarray(samples)
    hist, _, _ = np.histogram2d(
        samples_np[:, 0],
        samples_np[:, 1],
        bins=[x_edges, y_edges],
    )
    total = hist.sum()
    if total == 0.0:
        return float("nan")
    q = hist / total
    return float(np.sum(np.abs(q - ref)))


def run_target(target_name, target_index):
    logp, score_fn, mean_ref, cov_ref, bounds = get_target(target_name)
    x_edges, y_edges, grid_ref = _build_grid_reference(logp, bounds, GRID_L1_SIZE)

    xmin, xmax, ymin, ymax = _split_bounds(bounds)
    init_min = jnp.asarray([xmin, ymin], dtype=jnp.float32)
    init_max = jnp.asarray([xmax, ymax], dtype=jnp.float32)

    out_csv = os.path.join(OUT_DIR, f"{target_name}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target",
                "seed",
                "method",
                "iter",
                "grad_evals",
                "kernel_evals",
                "wall_s",
                "mu_err",
                "cov_err",
                "ess",
                "ksd",
                "mean_logp",
                "grid_l1",
                "nonfinite_frac",
                "m_used",
                "is_best",
            ]
        )

        for seed in SEEDS:
            seed_key = jax.random.fold_in(jax.random.PRNGKey(seed), target_index)
            init_key, chain_key, run_key = jax.random.split(seed_key, 3)

            init_particles = jax.random.uniform(
                init_key, (N_particles, 2), minval=init_min, maxval=init_max
            )
            x0_chain = jax.random.uniform(chain_key, (2,), minval=init_min, maxval=init_max)

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

            chain_buffers = {}
            grad_eval_counts = {}
            kernel_eval_counts = {}

            for name, (state, step_fn) in samplers.items():
                print(f"\n▶ {target_name} | seed {seed} | {name}")
                rng = run_key
                t0 = time.time()
                grad_eval_counts[name] = 0.0
                kernel_eval_counts[name] = 0.0
                best_ksd = None
                best_row = None
                best_iter = None
                last_iter = None
                bad_checks = 0
                check_count = 0
                nonfinite_streak = 0

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

                        mu_err_val = float(jnp.linalg.norm(samples.mean(0) - mean_ref))
                        cov_err_val = cov_error(samples, cov_ref)
                        ess_val = ess_1d(samples[:, 0])
                        ksd_val = ksd_rbf(samples, score_fn)
                        mlp_val = mean_log_prob(samples, logp)
                        grid_l1_val = _grid_l1(samples, x_edges, y_edges, grid_ref)
                        nonfinite_frac = float(info.get("nonfinite_frac", 0.0))
                        m_used = float(info.get("m_used", 0.0))

                        row = [
                            target_name,
                            seed,
                            name,
                            it,
                            grad_eval_counts[name],
                            kernel_eval_counts[name],
                            elapsed,
                            mu_err_val,
                            cov_err_val,
                            ess_val,
                            ksd_val,
                            mlp_val,
                            grid_l1_val,
                            nonfinite_frac,
                            m_used,
                            0,
                        ]
                        writer.writerow(row)
                        f.flush()
                        last_iter = it

                        if EARLY_STOP:
                            check_count += 1
                            if not math.isfinite(ksd_val):
                                nonfinite_streak += 1
                                bad_checks += 1
                            elif best_ksd is None or ksd_val < best_ksd:
                                best_ksd = ksd_val
                                best_row = row[:-1] + [1]
                                best_iter = it
                                bad_checks = 0
                                nonfinite_streak = 0
                            elif check_count >= EARLY_STOP_MIN_CHECKS and ksd_val > best_ksd * (1.0 + EARLY_STOP_TOL):
                                bad_checks += 1
                                nonfinite_streak = 0
                            else:
                                bad_checks = 0
                                nonfinite_streak = 0
                            if nonfinite_streak >= EARLY_STOP_NONFINITE_PATIENCE:
                                print(
                                    f"early stop: {target_name} | seed {seed} | {name} at iter {it} "
                                    f"(non-finite ksd for {nonfinite_streak} checkpoints)"
                                )
                                break
                            if check_count >= EARLY_STOP_MIN_CHECKS and bad_checks >= EARLY_STOP_PATIENCE:
                                best_str = f"{best_ksd:.3g}" if best_ksd is not None else "nan"
                                print(
                                    f"early stop: {target_name} | seed {seed} | {name} at iter {it} "
                                    f"(ksd {ksd_val:.3g} > best {best_str})"
                                )
                                break

                if best_row is not None and last_iter is not None and best_iter != last_iter:
                    writer.writerow(best_row)
                    f.flush()


def main():
    targets = RUN_TARGETS if RUN_TARGETS else list_targets()
    for i, tname in enumerate(targets):
        run_target(tname, i)


if __name__ == "__main__":
    main()
