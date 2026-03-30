# samplers.py  ────────────────────────────────────────────────────────────
import jax, jax.numpy as jnp
import optax
from functools import partial

# ───────────────────────────────────────────────────────────────────────────
# Pairwise kernels + repulsion terms
def kernel_and_rep(
    z,
    *,
    kernel="rbf",
    min_bw=1e-3,
    max_bw=1e3,
    bw_scale=1.0,
    rbf_scales=(0.25, 1.0, 4.0),
    imq_beta=0.5,
    imq_c=1.0,
):
    diff  = z[:, None, :] - z[None, :, :]          # (N,N,D)
    dist2 = jnp.sum(diff**2, axis=-1)              # (N,N)

    # --- ignore self-distances without boolean indexing ------------------
    big  = jnp.finfo(z.dtype).max
    dist2_nodiag = dist2 + jnp.eye(z.shape[0], dtype=z.dtype) * big

    median = jnp.median(dist2_nodiag)
    bw = jnp.clip(median * bw_scale, min_bw, max_bw)

    if kernel == "rbf":
        inv_h = 0.5 / bw
        K = jnp.exp(-dist2 * inv_h)
        rep = -2.0 * inv_h * (K[:, :, None] * diff).sum(axis=1)
        return K, rep

    if kernel == "rbf_multiscale":
        scales = jnp.asarray(rbf_scales, dtype=z.dtype)
        bws = jnp.clip(median * scales, min_bw, max_bw)
        inv_h = 0.5 / bws[:, None, None]
        K_parts = jnp.exp(-dist2[None, :, :] * inv_h)
        rep_parts = -2.0 * inv_h[:, :, :, None] * (K_parts[:, :, :, None] * diff[None, :, :, :])
        K = K_parts.mean(axis=0)
        rep = rep_parts.sum(axis=2).mean(axis=0)
        return K, rep

    if kernel == "imq":
        base = imq_c**2 + dist2 / bw
        K = base ** (-imq_beta)
        rep = -(2.0 * imq_beta / bw) * (K[:, :, None] * diff / base[:, :, None]).sum(axis=1)
        return K, rep

    raise ValueError(f"Unknown kernel '{kernel}'")


def rbf_kernel_and_rep(z, min_bw=1e-3, max_bw=1e3, bw_scale=1.0):
    return kernel_and_rep(z, kernel="rbf", min_bw=min_bw, max_bw=max_bw, bw_scale=bw_scale)

# ───────────────────────────────────────────────────────────────────────────
# 1) Vanilla Euler SVGD
def make_svgd_step(
    logprob_fn,
    step_size,
    *,
    bw_scale=1.0,
    kernel="rbf",
    rbf_scales=(0.25, 1.0, 4.0),
    imq_beta=0.5,
    imq_c=1.0,
):
    @jax.jit
    def step(z, _key):
        grads = jax.grad(lambda x: jnp.sum(logprob_fn(x)))(z)
        K, rep = kernel_and_rep(
            z,
            kernel=kernel,
            bw_scale=bw_scale,
            rbf_scales=rbf_scales,
            imq_beta=imq_beta,
            imq_c=imq_c,
        )
        z = z + step_size * (K @ grads + rep)
        return z, {
            "grad_evals": jnp.array(1.0, dtype=z.dtype),
            "kernel_evals": jnp.array(1.0, dtype=z.dtype),
        }
    return step

# ───────────────────────────────────────────────────────────────────────────
# 2) Strang-split / multi-rate SVGD
def make_strang_svgd_step(
    logprob_fn,
    step_size,
    M=4,
    *,
    bw_scale=1.0,
    kernel="rbf",
    rbf_scales=(0.25, 1.0, 4.0),
    imq_beta=0.5,
    imq_c=1.0,
):
    assert M % 2 == 0
    dt_R = step_size / M

    def repulsive(p, dt):
        _, rep = kernel_and_rep(
            p,
            kernel=kernel,
            bw_scale=bw_scale,
            rbf_scales=rbf_scales,
            imq_beta=imq_beta,
            imq_c=imq_c,
        )
        return p + dt * rep

    @jax.jit
    def step(z, _key):
        # half repulsive
        z = repulsive(z, 0.5 * dt_R)
        for _ in range(M // 2 - 1):
            z = repulsive(z, dt_R)
        # drift
        grads = jax.grad(lambda x: jnp.sum(logprob_fn(x)))(z)
        K, _ = kernel_and_rep(
            z,
            kernel=kernel,
            bw_scale=bw_scale,
            rbf_scales=rbf_scales,
            imq_beta=imq_beta,
            imq_c=imq_c,
        )
        z = z + step_size * (K @ grads)
        # second half repulsive
        for _ in range(M // 2 - 1):
            z = repulsive(z, dt_R)
        z = repulsive(z, 0.5 * dt_R)
        kernel_evals = jnp.array(float(M + 1), dtype=z.dtype)
        return z, {
            "grad_evals": jnp.array(1.0, dtype=z.dtype),
            "kernel_evals": kernel_evals,
        }
    return step

# ───────────────────────────────────────────────────────────────────────────
# 3) Stochastic Gradient Langevin Dynamics
def make_sgld_step(logprob_fn, lr):
    opt = optax.sgd(lr)

    def init(x0):
        return x0, opt.init(x0)           # returns (position, opt_state)

    @jax.jit
    def step(state, key):
        x, opt_state = state
        g = jax.grad(lambda y: jnp.sum(logprob_fn(y)))(x)
        noise = jax.random.normal(key, x.shape) * jnp.sqrt(2 * lr)
        updates, opt_state = opt.update(-g, opt_state)
        x = optax.apply_updates(x, updates) + noise
        return (x, opt_state), {
            "grad_evals": jnp.array(1.0, dtype=x.dtype),
            "kernel_evals": jnp.array(0.0, dtype=x.dtype),
        }
    return init, step

# ───────────────────────────────────────────────────────────────────────────
# 4) Stochastic Gradient Hamiltonian Monte Carlo
def make_sghmc_step(logprob_fn, lr, friction=0.05, mdecay=0.9):
    def init(x0):
        v = jnp.zeros_like(x0)
        return x0, v

    @jax.jit
    def step(state, key):
        x, v = state
        g = jax.grad(lambda y: jnp.sum(logprob_fn(y)))(x)
        noise = jax.random.normal(key, x.shape) * jnp.sqrt(2 * friction * lr)
        v = mdecay * v + lr * g + noise
        x = x + v
        return (x, v), {
            "grad_evals": jnp.array(1.0, dtype=x.dtype),
            "kernel_evals": jnp.array(0.0, dtype=x.dtype),
        }
    return init, step

# ───────────────────────────────────────────────────────────────────────────
# 5) Multirate-ratio SVGD  (your idea)
# 5)  Multirate-ratio SVGD  (stable)
# 5) Multirate-ratio SVGD  (stable version)
def make_multirate_svgd_step(
        logprob_fn,
        base_dt,                    # ← just once!
        *,
        m=4,
        grad_clip=50.0,             # clip ∇log p
        bw_scale=1.0,
        kernel="rbf",
        rbf_scales=(0.25, 1.0, 4.0),
        imq_beta=0.5,
        imq_c=1.0,
        debug=False):

    def _kernel_rep(z):
        K, rep = kernel_and_rep(
            z,
            kernel=kernel,
            bw_scale=bw_scale,
            rbf_scales=rbf_scales,
            imq_beta=imq_beta,
            imq_c=imq_c,
        )
        return K, rep

    def step(x, _key):
        z = x
        # clamp before kernel to avoid extreme distances causing inf/NaN
        z = jnp.clip(z, -1e2, 1e2)

        dt_rep = base_dt / m

        def repulsive(z_in):
            z_in = jnp.clip(z_in, -1e2, 1e2)
            _, rep = _kernel_rep(z_in)
            z_new = z_in + dt_rep * rep
            finite = jnp.isfinite(z_new)
            z_safe = jnp.where(finite, z_new, z_in)
            nonfinite = 1.0 - jnp.mean(finite)
            return z_safe, nonfinite

        # repulsion substeps (fast)
        if debug:
            m_int = int(m)
            rep_nonfinite_max = 0.0
            for _ in range(m_int):
                z, rep_nf = repulsive(z)
                rep_nonfinite_max = jnp.maximum(rep_nonfinite_max, rep_nf)
        else:
            def rep_loop(i, state):
                z_in, rep_nf_max = state
                z_out, rep_nf = repulsive(z_in)
                rep_nf_max = jnp.maximum(rep_nf_max, rep_nf)
                return z_out, rep_nf_max
            z, rep_nonfinite_max = jax.lax.fori_loop(0, m, rep_loop, (z, 0.0))

        # ---------- attractive drift (slow) ------------------------------
        g_raw = jax.grad(lambda y: jnp.sum(logprob_fn(y)))(z)
        g_raw = jnp.clip(g_raw, -grad_clip, grad_clip)
        K, _ = _kernel_rep(z)
        drift = K @ g_raw
        z = z + base_dt * drift

        # guard before final non-finite check
        z = jnp.clip(z, -1e4, 1e4)
        x_new = z

        # final NaN/Inf guard: keep prior values and surface diagnostics
        finite = jnp.isfinite(x_new)
        x_safe = jnp.where(finite, x_new, x)
        nonfinite_frac = 1.0 - jnp.mean(finite)
        info = {
            "nonfinite_frac": nonfinite_frac,
            "stiff_ratio": jnp.array(0.0, dtype=x.dtype),
            "m_used": jnp.array(float(m), dtype=x.dtype),
            "rep_nonfinite_max": rep_nonfinite_max,
            "grad_evals": jnp.array(1.0, dtype=x.dtype),
            "kernel_evals": jnp.array(float(m + 1), dtype=x.dtype),
        }
        return x_safe, info

    return step if debug else jax.jit(step)

# ───────────────────────────────────────────────────────────────────────────
# 6) Adaptive multirate SVGD (error-controlled drift substeps)
def make_adaptive_multirate_svgd_step(
        logprob_fn,
        base_dt,
        *,
        m_min=1,
        m_max=8,
        err_tol=1e-2,
        bw_scale=1.0,
        grad_clip=50.0,
        kernel="rbf",
        rbf_scales=(0.25, 1.0, 4.0),
        imq_beta=0.5,
        imq_c=1.0,
        debug=False):

    def _kernel_rep(z):
        K, rep = kernel_and_rep(
            z,
            kernel=kernel,
            bw_scale=bw_scale,
            rbf_scales=rbf_scales,
            imq_beta=imq_beta,
            imq_c=imq_c,
        )
        return K, rep

    def _rms(x):
        return jnp.sqrt(jnp.mean(x**2) + 1e-12)

    def _drift(z_in):
        g_raw = jax.grad(lambda y: jnp.sum(logprob_fn(y)))(z_in)
        g_raw = jnp.clip(g_raw, -grad_clip, grad_clip)
        K, _ = _kernel_rep(z_in)
        return K @ g_raw

    def _apply_drift(z_in, drift_in, dt):
        z_new = z_in + dt * drift_in
        finite = jnp.isfinite(z_new)
        z_safe = jnp.where(finite, z_new, z_in)
        nonfinite = 1.0 - jnp.mean(finite)
        return z_safe, nonfinite

    def step(x, _key):
        z = x
        z = jnp.clip(z, -1e2, 1e2)

        # Repulsion once per iteration (guard against nonfinite).
        _, rep0 = _kernel_rep(z)
        z_new = z + base_dt * rep0
        finite_rep = jnp.isfinite(z_new)
        z = jnp.where(finite_rep, z_new, z)
        rep_nonfinite_max = 1.0 - jnp.mean(finite_rep)

        # Error estimate: one full drift step vs two half steps.
        drift0 = _drift(z)
        z_full, nf_full = _apply_drift(z, drift0, base_dt)
        z_half, nf_half = _apply_drift(z, drift0, 0.5 * base_dt)
        drift_half = _drift(z_half)
        z_two, nf_two = _apply_drift(z_half, drift_half, 0.5 * base_dt)

        err = _rms(z_two - z_full)
        scale = _rms(z) + 1e-12
        err_rel = err / scale
        ratio = err_rel / err_tol
        ratio = jnp.where(jnp.isfinite(ratio), ratio, jnp.array(0.0, dtype=ratio.dtype))
        m_used = jnp.clip(jnp.ceil(jnp.sqrt(ratio)), m_min, m_max).astype(jnp.int32)
        m_used = jnp.where(jnp.isfinite(ratio), m_used, m_max)
        dt_drift = base_dt / m_used

        if debug:
            if int(m_used) <= 2:
                z = z_full if int(m_used) == 1 else z_two
                rep_nonfinite_max = jnp.maximum(
                    rep_nonfinite_max,
                    jnp.maximum(nf_full, jnp.maximum(nf_half, nf_two)),
                )
            else:
                z, nf0 = _apply_drift(z, drift0, dt_drift)
                rep_nonfinite_max = jnp.maximum(rep_nonfinite_max, nf0)
                for _ in range(1, int(m_used)):
                    drift = _drift(z)
                    z, drift_nf = _apply_drift(z, drift, dt_drift)
                    rep_nonfinite_max = jnp.maximum(rep_nonfinite_max, drift_nf)
        else:
            def _use_estimator(_):
                z_out = z_two
                nf_out = jnp.maximum(
                    rep_nonfinite_max,
                    jnp.maximum(nf_full, jnp.maximum(nf_half, nf_two)),
                )
                return z_out, nf_out

            def _use_loop(_):
                z0, nf0 = _apply_drift(z, drift0, dt_drift)
                nf0 = jnp.maximum(rep_nonfinite_max, nf0)

                def drift_loop(i, state):
                    z_in, nf_max = state

                    def _do_step(args):
                        z_local, nf_local = args
                        drift = _drift(z_local)
                        z_out, drift_nf = _apply_drift(z_local, drift, dt_drift)
                        nf_out = jnp.maximum(nf_local, drift_nf)
                        return z_out, nf_out

                    return jax.lax.cond(
                        i < m_used - 1,
                        _do_step,
                        lambda args: args,
                        (z_in, nf_max),
                    )

                z_out, nf_out = jax.lax.fori_loop(
                    0,
                    m_max - 1,
                    drift_loop,
                    (z0, nf0),
                )
                return z_out, nf_out

            z, rep_nonfinite_max = jax.lax.cond(
                m_used <= 2,
                _use_estimator,
                _use_loop,
                operand=None,
            )

        # guard before final non-finite check
        z = jnp.clip(z, -1e4, 1e4)
        x_new = z

        finite = jnp.isfinite(x_new)
        x_safe = jnp.where(finite, x_new, x)
        nonfinite_frac = 1.0 - jnp.mean(finite)
        grad_evals = jnp.where(
            m_used <= 2,
            jnp.array(2.0, dtype=x.dtype),
            m_used.astype(x.dtype) + jnp.array(1.0, dtype=x.dtype),
        )
        info = {
            "nonfinite_frac": nonfinite_frac,
            "stiff_ratio": ratio,
            "m_used": m_used.astype(jnp.float32),
            "rep_nonfinite_max": rep_nonfinite_max,
            "grad_evals": grad_evals,
            "kernel_evals": grad_evals + jnp.array(1.0, dtype=x.dtype),
        }
        return x_safe, info

    return step if debug else jax.jit(step)
