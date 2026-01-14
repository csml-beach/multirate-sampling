# target.py -----------------------------------------------------------------
import jax, jax.numpy as jnp

def make_gaussian50(key, d=50, log10_range=(0, -3)):
    """Return (logprob_fn, Σ, Σ^{-1/2}) for a random 50-dim Gaussian."""
    k1, k2 = jax.random.split(key)
    # random orthogonal matrix Q
    Q, _ = jnp.linalg.qr(jax.random.normal(k1, (d, d)))
    # eigenvalues log-spaced from 1 to 1e-3
    eigs = 10.0 ** jnp.linspace(*log10_range, d)
    Λ_sqrt = jnp.diag(jnp.sqrt(eigs))
    Σ = Q @ Λ_sqrt @ Λ_sqrt @ Q.T
    Σ_inv = Q @ jnp.diag(1.0 / eigs) @ Q.T

    def logprob(x):
        """log𝑝 up to an additive const (mean 0)."""
        q = (x * (Σ_inv @ x.T).T).sum(axis=-1)   # quadratic form
        return -0.5 * q

    return logprob, Σ, jnp.linalg.cholesky(Σ_inv)