from __future__ import annotations

import numpy as np


def _sample_groups(rng, n, n_groups, group_mode, zipf_alpha):
    if group_mode == "uniform":
        return rng.integers(0, n_groups, size=n, dtype=np.int32)
    if group_mode == "longtail":
        ranks = np.arange(1, n_groups + 1, dtype=np.float64)
        probs = ranks ** (-zipf_alpha)
        probs /= probs.sum()
        return rng.choice(n_groups, size=n, p=probs).astype(np.int32)
    raise ValueError(f"Unknown group_mode: {group_mode}")


def load_synthetic_hlr(
    *,
    seed=0,
    n_samples=200_000,
    n_features=256,
    n_groups=20_000,
    group_mode="longtail",
    feature_density=0.05,
    zipf_alpha=1.2,
    train_frac=0.8,
    prior_beta_std=1.0,
):
    rng = np.random.default_rng(seed)

    # Feature matrix with controllable sparsity.
    X = rng.normal(0.0, 1.0, size=(n_samples, n_features)).astype(np.float32)
    if feature_density < 1.0:
        keep = rng.random((n_samples, n_features)) < feature_density
        X *= keep.astype(np.float32)

    # Group assignments control random-intercept heterogeneity.
    g = _sample_groups(rng, n_samples, n_groups, group_mode, zipf_alpha)

    # Sparse-ish true global coefficients.
    beta_true = np.zeros(n_features, dtype=np.float32)
    n_active = max(8, n_features // 16)
    active = rng.choice(n_features, size=n_active, replace=False)
    beta_true[active] = rng.normal(0.0, prior_beta_std / np.sqrt(n_active), size=n_active).astype(np.float32)
    alpha_true = np.float32(rng.normal(0.0, 0.2))

    # Long-tail mode gets slightly stronger hierarchical variability.
    if group_mode == "longtail":
        log_tau_true = np.float32(rng.normal(-0.2, 0.35))
    else:
        log_tau_true = np.float32(rng.normal(-0.5, 0.25))
    tau_true = np.float32(np.exp(log_tau_true))

    z_true = rng.normal(0.0, 1.0, size=n_groups).astype(np.float32)
    u_true = tau_true * z_true

    logits = X @ beta_true + alpha_true + u_true[g]
    logits = np.clip(logits, -30.0, 30.0)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.float32)

    perm = rng.permutation(n_samples)
    n_train = int(train_frac * n_samples)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    g_train = g[train_idx]
    g_test = g[test_idx]

    # Standardization keeps optimization stable.
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-4, 1.0, std)
    X_train = ((X_train - mu) / std).astype(np.float32)
    X_test = ((X_test - mu) / std).astype(np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "g_train": g_train.astype(np.int32),
        "X_test": X_test,
        "y_test": y_test,
        "g_test": g_test.astype(np.int32),
        "meta": {
            "seed": int(seed),
            "group_mode": group_mode,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_groups": int(n_groups),
            "feature_density": float(feature_density),
            "zipf_alpha": float(zipf_alpha),
            "tau_true": float(tau_true),
            "alpha_true": float(alpha_true),
        },
    }
