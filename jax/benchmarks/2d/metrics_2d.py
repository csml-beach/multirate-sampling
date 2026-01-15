# metrics_2d.py  — moment errors + ESS + KSD -------------------------------
import jax.numpy as jnp
import numpy as np
from statsmodels.tsa.stattools import acf


def cov_error(samples, Sigma):
    """Frobenius norm between empirical and true covariance."""
    emp = jnp.cov(samples.T, bias=True)
    return float(jnp.linalg.norm(emp - Sigma))


def mean_log_prob(samples, logprob_fn):
    """Mean log-probability of samples under a target."""
    x = jnp.asarray(samples)
    if x.ndim == 1:
        x = x[None, :]
    return float(jnp.mean(logprob_fn(x)))


def ess_1d(chain):
    """Effective sample size of a 1-D chain using autocorrelation."""
    x = np.asarray(chain)
    ac = acf(x, fft=True, nlags=min(len(x) // 2, 100))
    pos = np.where(ac < 0)[0]
    T = pos[0] if pos.size else len(ac)
    return len(x) / (1 + 2 * ac[1:T].sum())


def ksd_rbf(samples, score_fn, bandwidth=None, min_bw=1e-3, max_bw=1e3):
    """Kernel Stein Discrepancy (RBF kernel)."""
    x = jnp.asarray(samples)
    if x.ndim == 1:
        x = x[None, :]
    n, d = x.shape
    if n < 2:
        return 0.0

    diff = x[:, None, :] - x[None, :, :]
    dist2 = jnp.sum(diff**2, axis=-1)

    if bandwidth is None:
        big = jnp.finfo(x.dtype).max
        dist2_nodiag = dist2 + jnp.eye(n, dtype=x.dtype) * big
        median = jnp.median(dist2_nodiag)
        bandwidth = jnp.clip(median, min_bw, max_bw)

    inv_bw = 1.0 / bandwidth
    k = jnp.exp(-0.5 * dist2 * inv_bw)

    score = score_fn(x)

    term1 = jnp.sum(score[:, None, :] * score[None, :, :], axis=-1) * k
    term2 = jnp.sum(score[:, None, :] * (-diff * inv_bw), axis=-1) * k
    term3 = jnp.sum(score[None, :, :] * (diff * inv_bw), axis=-1) * k
    term4 = (d * inv_bw - dist2 * inv_bw * inv_bw) * k

    ksd2 = jnp.sum(term1 + term2 + term3 + term4) / (n * n)
    return float(jnp.sqrt(jnp.maximum(ksd2, 0.0)))
