# benchmark_2d.py ----------------------------------------------------------
import jax
import jax.numpy as jnp
import time
import csv
import os
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
from metrics import cov_error, ess_1d, ksd_rbf, mean_log_prob


# ------------- configuration ---------------------------------------------
N_particles = 128
n_iter = 10_000
save_every = 100
chain_window = 1000

lr_svgd = 1e-3
lr_sgld = 1e-4
lr_sghmc = 1e-4

RUN_TARGETS = ["banana", "ring", "squiggly", "two_moons"]
RUN_METHODS = None  # set to a list of method names to filter

OUT_DIR = "metrics_2d"
os.makedirs(OUT_DIR, exist_ok=True)


def run_target(target_name, key):
    logp, score_fn, mean_ref, cov_ref, _bounds = get_target(target_name)

    init_particles = 0.5 * jax.random.normal(key, (N_particles, 2))
    x0_chain = 0.5 * jax.random.normal(key, (2,))

    samplers = {}
    samplers["multirate_svgd"] = (
        init_particles,
        make_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m=4,
            L_inv=None,
        ),
    )

    samplers["adaptive_multirate_svgd"] = (
        init_particles,
        make_adaptive_multirate_svgd_step(
            logp,
            base_dt=lr_svgd,
            m_min=1,
            m_max=16,
            err_tol=1e-2,
            L_inv=None,
        ),
    )

    samplers["vanilla_svgd"] = (
        init_particles,
        make_svgd_step(logp, lr_svgd),
    )

    samplers["strang_svgd"] = (
        init_particles,
        make_strang_svgd_step(logp, lr_svgd),
    )

    sgld_init_fn, sgld_step_fn = make_sgld_step(logp, lr_sgld)
    samplers["sgld"] = (sgld_init_fn(x0_chain), sgld_step_fn)

    sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, lr_sghmc)
    samplers["sghmc"] = (sghmc_init_fn(x0_chain), sghmc_step_fn)

    if RUN_METHODS is not None:
        samplers = {k: v for k, v in samplers.items() if k in RUN_METHODS}

    out_csv = os.path.join(OUT_DIR, f"{target_name}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target",
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
                "nonfinite_frac",
                "stiff_ratio",
                "m_used",
            ]
        )

        chain_buffers = {}
        grad_eval_counts = {}
        kernel_eval_counts = {}

        for name, (state, step_fn) in samplers.items():
            print(f"\n▶ {target_name} | {name}")
            rng = key
            t0 = time.time()
            grad_eval_counts[name] = 0.0
            kernel_eval_counts[name] = 0.0

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
                    nonfinite_frac = float(info.get("nonfinite_frac", 0.0))
                    stiff_ratio = float(info.get("stiff_ratio", 0.0))
                    m_used = float(info.get("m_used", 0.0))

                    writer.writerow(
                        [
                            target_name,
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
                            nonfinite_frac,
                            stiff_ratio,
                            m_used,
                        ]
                    )
                    f.flush()


def main():
    key = jax.random.PRNGKey(0)
    targets = RUN_TARGETS if RUN_TARGETS else list_targets()
    for i, tname in enumerate(targets):
        key, sub = jax.random.split(key)
        run_target(tname, sub)


if __name__ == "__main__":
    main()
