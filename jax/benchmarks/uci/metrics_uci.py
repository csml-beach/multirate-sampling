# metrics_uci.py ------------------------------------------------------------
import numpy as np
import jax.numpy as jnp
from statsmodels.tsa.stattools import acf


def accuracy(y_true, p_pred):
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    y_hat = (p_pred >= 0.5).astype(y_true.dtype)
    return float(np.mean(y_hat == y_true))


def nll(y_true, p_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    p_pred = np.clip(p_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred))
    return float(np.mean(loss))


def ece(y_true, p_pred, n_bins=10):
    y_true = np.asarray(y_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == (p_pred[mask] >= 0.5))
        conf = np.mean(p_pred[mask])
        ece_val += np.abs(acc - conf) * np.mean(mask)
    return float(ece_val)


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
