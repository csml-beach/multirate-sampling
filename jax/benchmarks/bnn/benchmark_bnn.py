# benchmark_bnn.py ---------------------------------------------------------
import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

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
from benchmarks.uci.datasets import load_breast_cancer, load_ionosphere, load_a5a
from benchmarks.uci.metrics_uci import accuracy, nll, ece, ess_1d, ksd_rbf, mean_log_prob


# ------------- configuration ---------------------------------------------
N_particles = 128
n_iter = 1_000
save_every = 50
chain_window = 200
SEEDS = range(5)

HIDDEN_DIM = 32
PRIOR_STD = 1.0

lr_svgd = 1e-3
lr_sgld = 1e-3
lr_sghmc = 1e-3
BW_SCALE = 0.1
ERR_TOL = 1e-2

COMPUTE_KSD = True
EARLY_STOP = True
EARLY_STOP_TOL = 0.1
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_CHECKS = 5

OUT_DIR = Path("metrics") / "bnn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "breast_cancer": load_breast_cancer,
    "ionosphere": load_ionosphere,
    "a5a": load_a5a,
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Run 1-hidden-layer BNN benchmarks.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names (or 'all').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name (overrides --datasets).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds (overrides SEEDS).",
    )
    return parser.parse_args()


def _select_datasets(arg_value):
    if arg_value.strip().lower() == "all":
        return list(DATASETS.keys())
    requested = [item.strip() for item in arg_value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {', '.join(unknown)}")
    return requested


def _as_batch(x):
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _unpack_params(theta, dim, hidden_dim):
    theta = _as_batch(theta)
    idx = 0
    w1 = theta[:, idx : idx + dim * hidden_dim].reshape(-1, dim, hidden_dim)
    idx += dim * hidden_dim
    b1 = theta[:, idx : idx + hidden_dim]
    idx += hidden_dim
    w2 = theta[:, idx : idx + hidden_dim].reshape(-1, hidden_dim, 1)
    idx += hidden_dim
    b2 = theta[:, idx : idx + 1]
    return w1, b1, w2, b2


def _bnn_logits(theta, X, hidden_dim):
    X = jnp.asarray(X)
    dim = X.shape[1]
    w1, b1, w2, b2 = _unpack_params(theta, dim, hidden_dim)
    hidden = jnp.tanh(jnp.einsum("nd,pdh->pnh", X, w1) + b1[:, None, :])
    logits = jnp.einsum("pnh,phk->pnk", hidden, w2).squeeze(-1) + b2
    return logits


def _make_logposterior(X, y, hidden_dim, prior_std):
    X = jnp.asarray(X)
    y = jnp.asarray(y)

    def logprob(theta):
        theta = _as_batch(theta)
        logits = _bnn_logits(theta, X, hidden_dim)
        loglik = jnp.sum(
            y * jax.nn.log_sigmoid(logits) + (1.0 - y) * jax.nn.log_sigmoid(-logits),
            axis=1,
        )
        logprior = -0.5 * jnp.sum(theta**2, axis=1) / (prior_std**2)
        return loglik + logprior

    return logprob


def _predictive_probs(samples, X, hidden_dim):
    theta = _as_batch(samples)
    logits = _bnn_logits(theta, X, hidden_dim)
    probs = jax.nn.sigmoid(logits)
    return jnp.mean(probs, axis=0)


def main():
    args = _parse_args()
    dataset_arg = args.dataset if args.dataset is not None else args.datasets
    dataset_names = _select_datasets(dataset_arg)
    seeds = SEEDS
    if args.seeds is not None:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
        if not seeds:
            raise ValueError("No seeds provided.")

    for dataset_name in dataset_names:
        loader = DATASETS[dataset_name]
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
                    "is_best",
                ]
            )

            for seed in seeds:
                data = loader(seed=seed)
                X_train = data["X_train"]
                y_train = data["y_train"]
                X_test = data["X_test"]
                y_test = data["y_test"]

                logp = _make_logposterior(X_train, y_train, HIDDEN_DIM, PRIOR_STD)
                score_fn = lambda w: jax.grad(lambda z: jnp.sum(logp(z)))(w)

                rng = jax.random.PRNGKey(seed)
                dim = X_train.shape[1]
                param_dim = dim * HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + 1

                init_particles = jax.random.normal(rng, (N_particles, param_dim)) * 0.1
                x0_chain = jax.random.normal(rng, (param_dim,)) * 0.1

                samplers = {
                    "multirate_svgd": (
                        init_particles,
                        make_multirate_svgd_step(
                            logp,
                            base_dt=lr_svgd,
                            m=4,
                            L_inv=None,
                            bw_scale=BW_SCALE,
                        ),
                    ),
                    "adaptive_multirate_svgd": (
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
                    ),
                    "vanilla_svgd": (
                        init_particles,
                        make_svgd_step(logp, lr_svgd, bw_scale=BW_SCALE),
                    ),
                    "strang_svgd": (
                        init_particles,
                        make_strang_svgd_step(logp, lr_svgd, bw_scale=BW_SCALE),
                    ),
                }

                sgld_init_fn, sgld_step_fn = make_sgld_step(logp, lr_sgld)
                samplers["sgld"] = (sgld_init_fn(x0_chain), sgld_step_fn)
                sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, lr_sghmc)
                samplers["sghmc"] = (sghmc_init_fn(x0_chain), sghmc_step_fn)

                chain_buffers = {}
                grad_eval_counts = {}
                kernel_eval_counts = {}

                for name, (state, step_fn) in samplers.items():
                    print(f"\n▶ {dataset_name} | seed {seed} | {name}")
                    step_rng = rng
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
                        step_rng, sub = jax.random.split(step_rng)
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

                            p_pred = _predictive_probs(samples, X_test, HIDDEN_DIM)
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
                                0,
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
                                    best_row = row[:-1] + [1]
                                    best_iter = it
                                    bad_checks = 0
                                elif (
                                    check_count >= EARLY_STOP_MIN_CHECKS
                                    and metric_val > best_ksd * (1.0 + EARLY_STOP_TOL)
                                ):
                                    bad_checks += 1
                                else:
                                    bad_checks = 0
                                if (
                                    check_count >= EARLY_STOP_MIN_CHECKS
                                    and bad_checks >= EARLY_STOP_PATIENCE
                                ):
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
