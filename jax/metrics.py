# metrics.py  — moment errors + ESS  (no KSD for now) -----------------------
import jax.numpy as jnp
import numpy as np
from statsmodels.tsa.stattools import acf

def mu_error(samples):
    """
    Compute the mean error for samples from a distribution.
    Assumes the true mean is zero (common for centered distributions).
    
    Args:
        samples: Array of shape (N, D) where N is number of samples, D is dimension
    
    Returns:
        float: L2 norm of the empirical mean vector
    """
    return float(jnp.linalg.norm(samples.mean(0)))

def cov_error(samples, Sigma):
    """
    Compute the covariance matrix error between empirical and true covariance.
    
    Args:
        samples: Array of shape (N, D) where N is number of samples, D is dimension
        Sigma: True covariance matrix of shape (D, D)
    
    Returns:
        float: Frobenius norm of the difference between empirical and true covariance
    """
    emp = jnp.cov(samples.T, bias=True)  # Empirical covariance matrix
    return float(jnp.linalg.norm(emp - Sigma))

def mean_log_prob(samples, logprob_fn):
    """
    Compute the mean log-probability of samples under the target.

    Args:
        samples: Array of shape (N, D) or (D,).
        logprob_fn: Callable mapping samples -> log-prob, shape (N,).
    Returns:
        float: Mean log-probability.
    """
    x = jnp.asarray(samples)
    if x.ndim == 1:
        x = x[None, :]
    return float(jnp.mean(logprob_fn(x)))

def ess_1d(chain):
    """
    Compute Effective Sample Size (ESS) for a 1-D chain using autocorrelation.
    Uses the initial-positive lag window method to estimate integrated autocorrelation time.
    
    The ESS quantifies how many independent samples the correlated chain is equivalent to.
    Higher ESS indicates better mixing and more efficient sampling.
    
    Args:
        chain: 1-D array of samples from a Markov chain
    
    Returns:
        float: Effective sample size
    """
    x = np.asarray(chain)
    # Compute autocorrelation function using FFT for efficiency
    ac = acf(x, fft=True, nlags=min(len(x)//2, 100))
    
    # Find first negative autocorrelation value
    pos = np.where(ac < 0)[0]
    # Use lag window up to first negative value (or full window if all positive)
    T = pos[0] if pos.size else len(ac)
    
    # ESS formula: N / (1 + 2 * sum of positive autocorrelations)
    return len(x) / (1 + 2*ac[1:T].sum())

def ksd_rbf(samples, score_fn, bandwidth=None, min_bw=1e-3, max_bw=1e3):
    """
    Kernel Stein Discrepancy (KSD) with an RBF kernel.

    Args:
        samples: Array of shape (N, D) or (D,).
        score_fn: Callable mapping samples -> score (grad log p), shape (N, D).
        bandwidth: Optional kernel bandwidth (h). Uses median heuristic if None.
    Returns:
        float: sqrt(KSD^2)
    """
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
