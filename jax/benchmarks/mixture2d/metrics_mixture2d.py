# metrics_mixture2d.py -----------------------------------------------------
import numpy as np
import jax.numpy as jnp
from statsmodels.tsa.stattools import acf


def assign_modes(samples, centers):
    x = jnp.asarray(samples)
    if x.ndim == 1:
        x = x[None, :]
    centers = jnp.asarray(centers)
    diff = x[:, None, :] - centers[None, :, :]
    dist2 = jnp.sum(diff**2, axis=-1)
    return np.asarray(jnp.argmin(dist2, axis=1))


def mode_stats(samples, centers, min_frac=0.05):
    idx = assign_modes(samples, centers)
    n_modes = centers.shape[0]
    counts = np.bincount(idx, minlength=n_modes).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0, 0.0, float("nan"), 0.0
    probs = counts / total
    coverage = float(np.mean(probs >= min_frac))
    if n_modes > 1:
        positive = probs > 0
        entropy = -np.sum(probs[positive] * np.log(probs[positive])) / np.log(n_modes)
    else:
        entropy = 0.0
    imbalance = float(np.std(probs))
    min_mass = float(np.min(probs))
    return coverage, float(entropy), imbalance, min_mass


def ess_1d(chain):
    x = np.asarray(chain)
    ac = acf(x, fft=True, nlags=min(len(x) // 2, 100))
    pos = np.where(ac < 0)[0]
    T = pos[0] if pos.size else len(ac)
    return len(x) / (1 + 2 * ac[1:T].sum())


def ksd_rbf(samples, score_fn, bandwidth=None, min_bw=1e-3, max_bw=1e3):
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


def mean_log_prob(samples, logprob_fn):
    x = jnp.asarray(samples)
    if x.ndim == 1:
        x = x[None, :]
    return float(jnp.mean(logprob_fn(x)))
