# benchmark_gauss50.py  ────────────────────────────────────────────────────
import jax, jax.numpy as jnp, time, csv, os, math
import sys
from pathlib import Path

JAX_DIR = Path(__file__).resolve().parents[2]
if str(JAX_DIR) not in sys.path:
    sys.path.insert(0, str(JAX_DIR))
from collections import deque
from tqdm import trange

from target_50d import make_gaussian50
from samplers import (make_svgd_step, make_strang_svgd_step,
                      make_sgld_step, make_sghmc_step,
                      make_multirate_svgd_step,
                      make_adaptive_multirate_svgd_step)
from metrics_50d  import mu_error, cov_error, ess_1d, ksd_rbf, mean_log_prob


# ------------- configuration ---------------------------------------------
N_particles = 128          # Number of particles for SVGD methods
n_iter      = 1_000        # Total number of sampling iterations
save_every  = 50           # Save metrics every N iterations
DEBUG_MULTIRATE = False    # Set True to debug multirate for a few steps
DEBUG_STEPS = 5
chain_window = 100        # Sliding window size for single-chain metrics
RUN_ONLY_ADAPTIVE = False   # Run only adaptive multirate for focused debugging
USE_WHITENING = False      # Apply whitening for SVGD-family methods
BW_SCALE = 0.5             # <1.0 strengthens repulsion (smaller bandwidth)
EARLY_STOP = True          # Stop when KSD worsens persistently
EARLY_STOP_TOL = 0.1      # Relative degradation allowed (5%)
EARLY_STOP_PATIENCE = 5    # Number of bad checkpoints before stopping
EARLY_STOP_MIN_CHECKS = 5  # Minimum checkpoints before applying early stop

# Learning rates for different methods
lr_svgd  = 1e-3            # SVGD learning rate
lr_sgld  = 1e-4            # SGLD learning rate (smaller for stability)
lr_sghmc = 1e-4            # SGHMC learning rate

out_csv = os.path.join("metrics", "50d", "metrics_gauss50.csv")
os.makedirs(os.path.dirname(out_csv), exist_ok=True)

# ----------- target distribution -----------------------------------------
key = jax.random.PRNGKey(0)
# Create a 50-dimensional Gaussian with random covariance structure
logp, Sigma, L_inv = make_gaussian50(key)     # keep the Cholesky inverse for multirate
score_fn = lambda x: jax.grad(lambda y: jnp.sum(logp(y)))(x)

# ----------- initial positions -------------------------------------------
# Starting point for single-chain methods (SGLD, SGHMC)
x0_chain = jax.random.uniform(key, (50,), minval=-4.0, maxval=4.0)
# Starting particles for ensemble methods (SVGD variants)
init_particles = jax.random.uniform(key, (N_particles, 50), minval=-4.0, maxval=4.0)

# ----------- build samplers ----------------------------------------------
samplers = {}

# SVGD variants - all use the same particle initialization for fair comparison
samplers["multirate_svgd"] = (
    init_particles,
    make_multirate_svgd_step(
        logp,
        base_dt=lr_svgd,   # Match SVGD step size for fair comparison
        m=4,               # Fixed repulsion substeps (IMEX-style)
        debug=DEBUG_MULTIRATE,
        L_inv=L_inv if USE_WHITENING else None,
        bw_scale=BW_SCALE)
)

samplers["adaptive_multirate_svgd"] = (
    init_particles,
    make_adaptive_multirate_svgd_step(
        logp,
        base_dt=lr_svgd,
        m_min=1, m_max=16,
        err_tol=1e-2,
        debug=DEBUG_MULTIRATE,
        L_inv=L_inv if USE_WHITENING else None,
        bw_scale=BW_SCALE)
)

if not DEBUG_MULTIRATE:
    samplers["vanilla_svgd"] = (
        init_particles,
        make_svgd_step(logp, lr_svgd, bw_scale=BW_SCALE)  # Standard SVGD with fixed step size
    )
    samplers["strang_svgd"] = (
        init_particles,
        make_strang_svgd_step(logp, lr_svgd, bw_scale=BW_SCALE)  # Strang splitting for SVGD dynamics
    )

    # Single-chain MCMC methods for comparison
    # SGLD - Stochastic Gradient Langevin Dynamics
    sgld_init_fn, sgld_step_fn = make_sgld_step(logp, lr_sgld)
    samplers["sgld"] = (
        sgld_init_fn(x0_chain),  # Initialize single chain
        sgld_step_fn
    )

    # SGHMC - Stochastic Gradient Hamiltonian Monte Carlo
    sghmc_init_fn, sghmc_step_fn = make_sghmc_step(logp, lr_sghmc)
    samplers["sghmc"] = (
        sghmc_init_fn(x0_chain),  # Initialize with position and momentum
        sghmc_step_fn
    )

if RUN_ONLY_ADAPTIVE:
    samplers = {"adaptive_multirate_svgd": samplers["adaptive_multirate_svgd"]}



# -------------- CSV header -----------------------------------------------
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    # Metrics: method name, iteration, wall time, mean error, covariance error, ESS
    writer.writerow([
        "method", "iter", "grad_evals", "kernel_evals", "wall_s",
        "mu_err", "cov_err", "ess", "ksd", "mean_logp",
        "nonfinite_frac", "stiff_ratio", "m_used",
    ])

    # -------------- main benchmark loop ----------------------------------
    chain_buffers = {}

    grad_eval_counts = {}
    kernel_eval_counts = {}
    for name, (state, step_fn) in samplers.items():
        print(f"\n▶ {name}")
        rng = key
        t0  = time.time()  # Start timing for this method
        grad_eval_counts[name] = 0.0
        kernel_eval_counts[name] = 0.0
        best_ksd = None
        bad_checks = 0
        check_count = 0
        if name in {"sgld", "sghmc"}:
            chain_buffers[name] = deque(maxlen=chain_window)
        
        for it in trange(1, n_iter + 1):
            rng, sub = jax.random.split(rng)  # Generate fresh randomness
            state, info = step_fn(state, sub)    # Take one sampling step
            grad_eval_counts[name] += float(info.get("grad_evals", 1.0))
            kernel_eval_counts[name] += float(info.get("kernel_evals", 0.0))
            if DEBUG_MULTIRATE and name == "multirate_svgd" and it <= DEBUG_STEPS:
                print(f"debug multirate iter {it}: {info}")

            # Periodically compute and save metrics
            if it % save_every == 0:
                elapsed = time.time() - t0
                # Extract samples (handle both particle arrays and chain states)
                if name in {"sgld", "sghmc"}:
                    chain_buffers[name].append(state[0])
                    samples = jnp.stack(list(chain_buffers[name]))
                else:
                    samples = state if isinstance(state, jnp.ndarray) else state[0]
                grad_evals = grad_eval_counts[name]
                kernel_evals = kernel_eval_counts[name]
                
                # Compute quality metrics
                mu_err_val  = mu_error(samples)        # Mean estimation error
                cov_err_val = cov_error(samples, Sigma)  # Covariance error
                ess_val     = ess_1d(samples[:, 0] if samples.ndim == 2 else samples)  # ESS of first component
                ksd_val     = ksd_rbf(samples, score_fn)
                mlp_val     = mean_log_prob(samples, logp)
                nonfinite_frac = float(info.get("nonfinite_frac", 0.0))
                stiff_ratio = float(info.get("stiff_ratio", 0.0))
                m_used = float(info.get("m_used", 0.0))
                
                # Write to CSV for later analysis
                writer.writerow([
                    name, it, grad_evals, kernel_evals, elapsed,
                    mu_err_val, cov_err_val, ess_val, ksd_val, mlp_val,
                    nonfinite_frac, stiff_ratio, m_used,
                ])
                f.flush()  # Ensure data is written immediately
                if EARLY_STOP:
                    check_count += 1
                    if not math.isfinite(ksd_val):
                        bad_checks += 1
                    elif best_ksd is None or ksd_val < best_ksd:
                        best_ksd = ksd_val
                        bad_checks = 0
                    elif check_count >= EARLY_STOP_MIN_CHECKS and ksd_val > best_ksd * (1.0 + EARLY_STOP_TOL):
                        bad_checks += 1
                    else:
                        bad_checks = 0
                    if check_count >= EARLY_STOP_MIN_CHECKS and bad_checks >= EARLY_STOP_PATIENCE:
                        print(
                            f"early stop: {name} at iter {it} "
                            f"(ksd {ksd_val:.3g} > best {best_ksd:.3g})"
                        )
                        break

            if DEBUG_MULTIRATE and name == "multirate_svgd" and it >= DEBUG_STEPS:
                break

# For SVGD methods, you might want to collect particle history
# and use all particles across iterations for mean estimation
# Or compare ESS/second as the primary metric
