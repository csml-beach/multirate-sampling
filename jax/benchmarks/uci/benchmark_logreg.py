# benchmark_logreg.py ------------------------------------------------------
import os
import sys
import time
from pathlib import Path
from collections import deque
import csv
import math

import jax
import jax.numpy as jnp
import numpy as np

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))

from samplers import (
    make_svgd_step,
    make_strang_svgd_step,
    make_sgld_step,
    make_sghmc_step,
    make_multirate_svgd_step,
    make_adaptive_multirate_svgd_step,
)
from benchmarks.uci.datasets import load_breast_cancer, load_ionosphere
from benchmarks.uci.metrics_uci import accuracy, nll, ece, ess_1d, ksd_rbf, mean_log_prob


# ------------- configuration ---------------------------------------------
N_particles = 128
n_iter = 1_000
save_every = 20
chain_window = 200
SEEDS = range(5)

lr_svgd = 1e-2
lr_sgld = 1e-2
lr_sghmc = 1e-2
BW_SCALE = 0.1
ERR_TOL = 1e-2
PRIOR_STD = 3.0

COMPUTE_KSD = False
EARLY_STOP = True          # Stop when KSD worsens persistently
EARLY_STOP_TOL = 0.05      # Relative degradation allowed (5%)
EARLY_STOP_PATIENCE = 10    # Number of bad checkpoints before stopping
EARLY_STOP_MIN_CHECKS = 10  # Minimum checkpoints before applying early stop

OUT_DIR = Path("metrics") / "uci"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASETS = {
    "breast_cancer": load_breast_cancer,
    "ionosphere": load_ionosphere,
}


def _as_batch(x):
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _make_logposterior(X, y, prior_std):
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    n = X.shape[0]

    def logprob(w):
        w = _as_batch(w)
        logits = w @ X.T  # (n_particles, n)
        loglik = jnp.sum(
            y * jax.nn.log_sigmoid(logits) + (1.0 - y) * jax.nn.log_sigmoid(-logits),
            axis=1,
        )
        logprior = -0.5 * jnp.sum(w**2, axis=1) / (prior_std**2)
        return loglik + logprior

    return logprob


def _predictive_probs(samples, X):
    w = jnp.asarray(samples)
    if w.ndim == 1:
        w = w[None, :]
    logits = w @ jnp.asarray(X).T
    probs = jax.nn.sigmoid(logits)
    return jnp.mean(probs, axis=0)


def main():
    for dataset_name, loader in DATASETS.items():
        out_csv = OUT_DIR / f"{dataset_name}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "dataset",
                    "seed",
                    "method",
                    "iter",
                    "grad_evals",
                    "kernel_evals",
                    "wall_s",
                    "acc",
                    "nll",
                    "ece",
                    "ess",
                    "ksd",
                    "mean_logp",
                ]
            )

            for seed in SEEDS:
                data = loader(seed=seed)
                X_train = data["X_train"]
                y_train = data["y_train"]
                X_test = data["X_test"]
                y_test = data["y_test"]

                logp = _make_logposterior(X_train, y_train, PRIOR_STD)
                score_fn = lambda w: jax.grad(lambda z: jnp.sum(logp(z)))(w)

                key = jax.random.PRNGKey(seed)
                dim = X_train.shape[1]

                init_particles = jax.random.normal(key, (N_particles, dim)) * 0.1
                x0_chain = jax.random.normal(key, (dim,)) * 0.1

                samplers = {}
                samplers["multirate_svgd"] = (
                    init_particles,
                    make_multirate_svgd_step(logp, base_dt=lr_svgd, m=4, L_inv=None, bw_scale=BW_SCALE),
                )
                samplers["adaptive_multirate_svgd"] = (
                    init_particles,
                    make_adaptive_multirate_svgd_step(
                        logp,
                        base_dt=lr_svgd,
                        m_min=1,
                        m_max=8,
                        err_tol=ERR_TOL,
                        L_inv=None,
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

                chain_buffers = {}
                grad_eval_counts = {}
                kernel_eval_counts = {}

                for name, (state, step_fn) in samplers.items():
                    print(f"\n▶ {dataset_name} | seed {seed} | {name}")
                    rng = key
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

                    for it in range(1, n_iter + 1):
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

                            p_pred = _predictive_probs(samples, X_test)
                            acc_val = accuracy(y_test, p_pred)
                            nll_val = nll(y_test, p_pred)
                            ece_val = ece(y_test, p_pred)
                            ess_val = ess_1d(samples[:, 0] if samples.ndim == 2 else samples)
                            ksd_val = ksd_rbf(samples, score_fn) if COMPUTE_KSD else float("nan")
                            mlp_val = mean_log_prob(samples, logp)

                            row = [
                                dataset_name,
                                seed,
                                name,
                                it,
                                grad_eval_counts[name],
                                kernel_eval_counts[name],
                                elapsed,
                                acc_val,
                                nll_val,
                                ece_val,
                                ess_val,
                                ksd_val,
                                mlp_val,
                            ]
                            writer.writerow(row)
                            f.flush()
                            last_iter = it
                            if EARLY_STOP:
                                check_count += 1
                                metric_val = ksd_val if COMPUTE_KSD else nll_val
                                if not math.isfinite(metric_val):
                                    bad_checks += 1
                                elif best_ksd is None or metric_val < best_ksd:
                                    best_ksd = metric_val
                                    best_row = row
                                    best_iter = it
                                    bad_checks = 0
                                elif check_count >= EARLY_STOP_MIN_CHECKS and metric_val > best_ksd * (1.0 + EARLY_STOP_TOL):
                                    bad_checks += 1
                                else:
                                    bad_checks = 0
                                if check_count >= EARLY_STOP_MIN_CHECKS and bad_checks >= EARLY_STOP_PATIENCE:
                                    best_str = f"{best_ksd:.3g}" if best_ksd is not None else "nan"
                                    metric_name = "ksd" if COMPUTE_KSD else "nll"
                                    print(
                                        f"early stop: {dataset_name} | seed {seed} | {name} at iter {it} "
                                        f"({metric_name} {metric_val:.3g} > best {best_str})"
                                    )
                                    break
                    if best_row is not None and last_iter is not None and best_iter != last_iter:
                        writer.writerow(best_row)
                        f.flush()


if __name__ == "__main__":
    main()
