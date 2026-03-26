# targets_2d.py ------------------------------------------------------------
import os
import jax
import jax.numpy as jnp
import numpy as np


_CACHE_DIR = os.path.join(os.path.dirname(__file__), "targets_2d_cache")
_CACHE_VERSION = "v2"


def _as_batch(x):
    x = jnp.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def _banana_logp(x, b=0.03, v=100.0, sigma_y=1.0):
    x = _as_batch(x)
    x1, x2 = x[:, 0], x[:, 1]
    # Canonical twisted-Gaussian banana: x1 has variance v and x2 is warped by b*x1^2.
    y = x2 + b * (x1**2 - v)
    return -0.5 * (x1**2 / v + (y**2) / (sigma_y**2))


def _ring_logp(x, r0=2.5, sigma=0.3):
    x = _as_batch(x)
    r = jnp.sqrt(jnp.sum(x**2, axis=-1) + 1e-12)
    return -0.5 * ((r - r0) ** 2) / (sigma**2)


def _squiggly_logp(x, amp=1.0, freq=3.0, sigma_x=2.0, sigma_y=0.3):
    x = _as_batch(x)
    x1, x2 = x[:, 0], x[:, 1]
    center = amp * jnp.sin(freq * x1)
    return -0.5 * (x1**2) / (sigma_x**2) - 0.5 * ((x2 - center) ** 2) / (sigma_y**2)


def _two_moons_logp(x, r=2.0, sigma=0.2, shift=1.0, offset=-0.5, sigma_x=2.0):
    x = _as_batch(x)
    x1, x2 = x[:, 0], x[:, 1]

    rad1 = jnp.sqrt(jnp.maximum(r**2 - x1**2, 0.0))
    rad2 = jnp.sqrt(jnp.maximum(r**2 - (x1 - shift) ** 2, 0.0))
    y1 = rad1
    y2 = -rad2 + offset

    l1 = -0.5 * ((x2 - y1) ** 2) / (sigma**2)
    l2 = -0.5 * ((x2 - y2) ** 2) / (sigma**2)
    envelope = -0.5 * ((x1 - 0.5 * shift) ** 2) / (sigma_x**2)
    return jnp.logaddexp(l1, l2) + envelope


def _funnel_logp(x, sigma_v=3.0):
    x = _as_batch(x)
    v = x[:, 0]
    y = x[:, 1]
    return -0.5 * (v**2) / (sigma_v**2) - 0.5 * (v + (y**2) / jnp.exp(v))


_TARGETS = {
    "banana": {
        "logp": _banana_logp,
        "bounds": ((-20.0, 20.0), (-12.0, 8.0)),
    },
    "ring": {
        "logp": _ring_logp,
        "bounds": (-4.0, 4.0),
    },
    "squiggly": {
        "logp": _squiggly_logp,
        "bounds": (-6.0, 6.0),
    },
    "two_moons": {
        "logp": _two_moons_logp,
        "bounds": (-6.0, 6.0),
    },
    "funnel": {
        "logp": _funnel_logp,
        "bounds": (-6.0, 6.0),
    },
}


def _split_bounds(bounds):
    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and np.isscalar(bounds[0])
        and np.isscalar(bounds[1])
    ):
        xmin, xmax = float(bounds[0]), float(bounds[1])
        ymin, ymax = xmin, xmax
        return xmin, xmax, ymin, ymax

    if (
        isinstance(bounds, tuple)
        and len(bounds) == 2
        and isinstance(bounds[0], tuple)
        and isinstance(bounds[1], tuple)
        and len(bounds[0]) == 2
        and len(bounds[1]) == 2
    ):
        xmin, xmax = float(bounds[0][0]), float(bounds[0][1])
        ymin, ymax = float(bounds[1][0]), float(bounds[1][1])
        return xmin, xmax, ymin, ymax

    raise ValueError(f"Invalid bounds format: {bounds!r}")


def list_targets():
    return sorted(_TARGETS.keys())


def _estimate_reference_moments(logp_fn, bounds, grid_size=400):
    xmin, xmax, ymin, ymax = _split_bounds(bounds)
    xs = np.linspace(xmin, xmax, grid_size)
    ys = np.linspace(ymin, ymax, grid_size)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    logp = np.asarray(logp_fn(jnp.asarray(pts)))
    logp = logp - np.max(logp)
    w = np.exp(logp)
    w_sum = np.sum(w)
    if w_sum == 0.0:
        raise ValueError("Reference moment estimation failed: zero total weight.")

    mean = np.sum(w[:, None] * pts, axis=0) / w_sum
    diff = pts - mean
    cov = np.sum(w[:, None, None] * (diff[:, :, None] * diff[:, None, :]), axis=0) / w_sum
    return mean, cov


def get_target(name, grid_size=400, cache=True):
    if name not in _TARGETS:
        raise ValueError(f"Unknown target '{name}'. Available: {list_targets()}")

    spec = _TARGETS[name]
    logp_fn = spec["logp"]
    bounds = spec["bounds"]

    cache_path = os.path.join(_CACHE_DIR, f"{name}_grid{grid_size}_{_CACHE_VERSION}.npz")
    if cache and os.path.exists(cache_path):
        data = np.load(cache_path)
        mean = data["mean"]
        cov = data["cov"]
    else:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        mean, cov = _estimate_reference_moments(logp_fn, bounds, grid_size=grid_size)
        if cache:
            np.savez(cache_path, mean=mean, cov=cov)

    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)

    def score_fn(x):
        return jax.grad(lambda y: jnp.sum(logp_fn(y)))(x)

    return logp_fn, score_fn, mean, cov, bounds
