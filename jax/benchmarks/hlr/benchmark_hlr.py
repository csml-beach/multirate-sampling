import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path
import sys

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
from benchmarks.hlr.datasets_hlr import load_synthetic_hlr
from benchmarks.uci.metrics_uci import accuracy, nll, ece, ess_1d, ksd_rbf, mean_log_prob


# ------------- defaults ---------------------------------------------------
N_PARTICLES = 32
N_ITER = 1_000
SAVE_EVERY = 20
CHAIN_WINDOW = 200
SEEDS = range(3)

LR_MR = 3e-3
LR_ADAPT = 3e-3
LR_VANILLA = 3e-4
LR_STRANG = 3e-4
LR_SGLD = 3e-4
LR_SGHMC = 3e-4
BW_SCALE = 0.1
ERR_TOL = 1e-2

PRIOR_BETA_STD = 1.0
PRIOR_ALPHA_STD = 2.0
PRIOR_LOGTAU_MEAN = -0.5
PRIOR_LOGTAU_STD = 1.0

COMPUTE_KSD = False

# NLL-only early stop (as requested)
EARLY_STOP = True
EARLY_STOP_TOL = 0.02
EARLY_STOP_PATIENCE = 8
EARLY_STOP_MIN_CHECKS = 8
EARLY_STOP_NONFINITE_PATIENCE = 2

OUT_DIR = Path("metrics") / "hlr"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _as_batch(x):
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _unpack_theta(theta, n_features, n_groups):
    theta = _as_batch(theta)
    idx = 0
    beta = theta[:, idx : idx + n_features]
    idx += n_features
    alpha = theta[:, idx : idx + 1]
    idx += 1
    z = theta[:, idx : idx + n_groups]
    idx += n_groups
    log_tau = theta[:, idx : idx + 1]
    return beta, alpha, z, log_tau


def _make_logposterior(X, y, g, n_features, n_groups):
    X = jnp.asarray(X)
    y = jnp.asarray(y)
    g = jnp.asarray(g, dtype=jnp.int32)

    def logprob(theta):
        beta, alpha, z, log_tau = _unpack_theta(theta, n_features, n_groups)
        tau = jnp.exp(log_tau)
        u = z * tau

        logits = beta @ X.T + alpha + u[:, g]
        logits = jnp.clip(logits, -40.0, 40.0)
        loglik = jnp.sum(
            y * jax.nn.log_sigmoid(logits) + (1.0 - y) * jax.nn.log_sigmoid(-logits),
            axis=1,
        )

        lp_beta = -0.5 * jnp.sum((beta / PRIOR_BETA_STD) ** 2, axis=1)
        lp_alpha = -0.5 * jnp.sum((alpha / PRIOR_ALPHA_STD) ** 2, axis=1)
        lp_z = -0.5 * jnp.sum(z**2, axis=1)
        lp_log_tau = -0.5 * jnp.sum(((log_tau - PRIOR_LOGTAU_MEAN) / PRIOR_LOGTAU_STD) ** 2, axis=1)

        return loglik + lp_beta + lp_alpha + lp_z + lp_log_tau

    return logprob


def _predictive_probs(samples, X, g, n_features, n_groups):
    X = jnp.asarray(X)
    g = jnp.asarray(g, dtype=jnp.int32)
    theta = _as_batch(samples)
    beta, alpha, z, log_tau = _unpack_theta(theta, n_features, n_groups)
    tau = jnp.exp(log_tau)
    u = z * tau
    logits = beta @ X.T + alpha + u[:, g]
    probs = jax.nn.sigmoid(logits)
    return jnp.mean(probs, axis=0)


def _safe_ess(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size < 3:
        return float("nan")
    if not np.all(np.isfinite(x)):
        return float("nan")
    if float(np.std(x)) < 1e-12:
        return float("nan")
    try:
        val = float(ess_1d(x))
        if not np.isfinite(val):
            return float("nan")
        return val
    except Exception:
        return float("nan")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic hierarchical logistic-regression benchmarks.")
    parser.add_argument(
        "--group-mode",
        type=str,
        default="longtail",
        choices=["longtail", "uniform", "both"],
        help="Group assignment mode. Default is longtail.",
    )
    parser.add_argument("--n-samples", type=int, default=200_000)
    parser.add_argument("--n-features", type=int, default=256)
    parser.add_argument("--n-groups", type=int, default=20_000)
    parser.add_argument("--feature-density", type=float, default=0.05)
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated list, e.g. 0,1,2")
    parser.add_argument("--iters", type=int, default=N_ITER)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--particles", type=int, default=N_PARTICLES)
    parser.add_argument("--lr-mr", type=float, default=LR_MR)
    parser.add_argument("--lr-adapt", type=float, default=LR_ADAPT)
    parser.add_argument("--lr-vanilla", type=float, default=LR_VANILLA)
    parser.add_argument("--lr-strang", type=float, default=LR_STRANG)
    parser.add_argument("--lr-sgld", type=float, default=LR_SGLD)
    parser.add_argument("--lr-sghmc", type=float, default=LR_SGHMC)
    return parser.parse_args()


def _select_group_modes(group_mode):
    if group_mode == "both":
        return ["longtail", "uniform"]
    return [group_mode]


def _select_seeds(seeds_arg):
    if seeds_arg is None:
        return list(SEEDS)
    vals = [int(item.strip()) for item in seeds_arg.split(",") if item.strip()]
    if not vals:
        raise ValueError("No seeds provided.")
    return vals


def main():
    args = _parse_args()
    group_modes = _select_group_modes(args.group_mode)
    seeds = _select_seeds(args.seeds)

    for group_mode in group_modes:
        out_csv = OUT_DIR / f"{group_mode}.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "scenario",
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
                data = load_synthetic_hlr(
                    seed=seed,
                    n_samples=args.n_samples,
                    n_features=args.n_features,
                    n_groups=args.n_groups,
                    group_mode=group_mode,
                    feature_density=args.feature_density,
                    zipf_alpha=args.zipf_alpha,
                )
                X_train = data["X_train"]
                y_train = data["y_train"]
                g_train = data["g_train"]
                X_test = data["X_test"]
                y_test = data["y_test"]
                g_test = data["g_test"]

                n_features = X_train.shape[1]
                n_groups = args.n_groups
                dim = n_features + 1 + n_groups + 1

                logp = _make_logposterior(X_train, y_train, g_train, n_features, n_groups)
                score_fn = lambda w: jax.grad(lambda z: jnp.sum(logp(z)))(w)

                key = jax.random.PRNGKey(seed)
                init_particles = jax.random.normal(key, (args.particles, dim)) * 0.05
                x0_chain = jax.random.normal(key, (dim,)) * 0.05

                samplers = {
                    "multirate_svgd": (
                        init_particles,
                        make_multirate_svgd_step(logp, base_dt=args.lr_mr, m=4, bw_scale=BW_SCALE),
                    ),
                    "adaptive_multirate_svgd": (
                        init_particles,
                        make_adaptive_multirate_svgd_step(
                            logp,
                            base_dt=args.lr_adapt,
                            m_min=1,
                            m_max=8,
                            err_tol=ERR_TOL,
                            bw_scale=BW_SCALE,
                        ),
                    ),
                    "vanilla_svgd": (
                        init_particles,
                        make_svgd_step(logp, args.lr_vanilla, bw_scale=BW_SCALE),
                    ),
                    "strang_svgd": (
                        init_particles,
                        make_strang_svgd_step(logp, args.lr_strang, bw_scale=BW_SCALE),
                    ),
                }

                sgld_init_fn, sgld_step_fn = make_sgld_step(logp, args.lr_sgld)
                samplers["sgld"] = (sgld_init_fn(x0_chain), sgld_step_fn)
                sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, args.lr_sghmc)
                samplers["sghmc"] = (sghmc_init_fn(x0_chain), sghmc_step_fn)

                chain_buffers = {}
                grad_eval_counts = {}
                kernel_eval_counts = {}

                for name, (state, step_fn) in samplers.items():
                    print(f"\n▶ hlr:{group_mode} | seed {seed} | {name}")
                    rng = key
                    t0 = time.time()
                    grad_eval_counts[name] = 0.0
                    kernel_eval_counts[name] = 0.0
                    best_nll = None
                    best_row = None
                    best_iter = None
                    last_iter = None
                    bad_checks = 0
                    check_count = 0
                    nonfinite_streak = 0

                    if name in {"sgld", "sghmc"}:
                        chain_buffers[name] = deque(maxlen=CHAIN_WINDOW)

                    for it in range(1, args.iters + 1):
                        rng, sub = jax.random.split(rng)
                        state, info = step_fn(state, sub)
                        grad_eval_counts[name] += float(info.get("grad_evals", 1.0))
                        kernel_eval_counts[name] += float(info.get("kernel_evals", 0.0))

                        if it % args.save_every == 0:
                            elapsed = time.time() - t0

                            if name in {"sgld", "sghmc"}:
                                chain_buffers[name].append(state[0])
                                samples = jnp.stack(list(chain_buffers[name]))
                            else:
                                samples = state if isinstance(state, jnp.ndarray) else state[0]

                            p_pred = _predictive_probs(samples, X_test, g_test, n_features, n_groups)
                            acc_val = accuracy(y_test, p_pred)
                            nll_val = nll(y_test, p_pred)
                            ece_val = ece(y_test, p_pred)
                            ess_series = samples[:, 0] if samples.ndim == 2 else samples
                            ess_val = _safe_ess(ess_series)
                            ksd_val = ksd_rbf(samples, score_fn) if COMPUTE_KSD else float("nan")
                            mlp_val = mean_log_prob(samples, logp)

                            row = [
                                group_mode,
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
                                metric_val = nll_val
                                if not math.isfinite(metric_val):
                                    nonfinite_streak += 1
                                    bad_checks += 1
                                elif best_nll is None or metric_val < best_nll:
                                    best_nll = metric_val
                                    best_row = row[:-1] + [1]
                                    best_iter = it
                                    bad_checks = 0
                                    nonfinite_streak = 0
                                elif check_count >= EARLY_STOP_MIN_CHECKS and metric_val > best_nll * (1.0 + EARLY_STOP_TOL):
                                    bad_checks += 1
                                    nonfinite_streak = 0
                                else:
                                    bad_checks = 0
                                    nonfinite_streak = 0

                                if nonfinite_streak >= EARLY_STOP_NONFINITE_PATIENCE:
                                    print(
                                        f"early stop: hlr:{group_mode} | seed {seed} | {name} at iter {it} "
                                        f"(non-finite nll for {nonfinite_streak} checkpoints)"
                                    )
                                    break

                                if check_count >= EARLY_STOP_MIN_CHECKS and bad_checks >= EARLY_STOP_PATIENCE:
                                    best_str = f"{best_nll:.3g}" if best_nll is not None else "unavailable"
                                    print(
                                        f"early stop: hlr:{group_mode} | seed {seed} | {name} at iter {it} "
                                        f"(nll {metric_val:.3g} > best {best_str})"
                                    )
                                    break

                    if best_row is not None and last_iter is not None and best_iter != last_iter:
                        writer.writerow(best_row)
                        f.flush()


if __name__ == "__main__":
    main()
