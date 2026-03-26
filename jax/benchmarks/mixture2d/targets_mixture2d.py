# targets_mixture2d.py -----------------------------------------------------
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


def _make_ring_centers(n_modes=8, radius=4.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n_modes, endpoint=False)
    centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return centers


def _make_grid_centers(side=2, spacing=3.0):
    coords = np.linspace(-(side - 1) / 2.0, (side - 1) / 2.0, side) * spacing
    xx, yy = np.meshgrid(coords, coords, indexing="xy")
    centers = np.stack([xx.ravel(), yy.ravel()], axis=1)
    return centers


def _build_target(name):
    if name == "mix8":
        centers = _make_ring_centers(n_modes=8, radius=4.0)
    else:
        raise ValueError(f"Unknown mixture target '{name}'")

    centers = jnp.asarray(centers, dtype=jnp.float32)
    n_modes = centers.shape[0]
    weights = jnp.ones((n_modes,), dtype=jnp.float32) / n_modes
    sigma = jnp.array([0.35, 0.7], dtype=jnp.float32)
    inv_var = 1.0 / (sigma**2)
    log_norm = -0.5 * (jnp.log(2.0 * jnp.pi * sigma**2).sum())

    def logp(x):
        x = jnp.asarray(x)
        if x.ndim == 1:
            x = x[None, :]
        diff = x[:, None, :] - centers[None, :, :]
        quad = -0.5 * jnp.sum(diff**2 * inv_var, axis=-1)
        log_probs = log_norm + quad + jnp.log(weights)
        return logsumexp(log_probs, axis=1)

    spread = 4.0 * float(jnp.max(sigma))
    min_c = float(jnp.min(centers)) - spread
    max_c = float(jnp.max(centers)) + spread
    bounds = (min_c, max_c)
    return logp, centers, bounds


def get_target(name):
    return _build_target(name)


def list_targets():
    return ["mix8"]
