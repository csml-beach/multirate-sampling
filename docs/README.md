# Multirate Sampling (SVGD-focused)

## Purpose
This repository explores multirate variants of particle-based sampling, with an initial focus on SVGD-style dynamics. The core idea is to split or reweight the repulsive (kernel) and attractive (log-density gradient) components of SVGD so they can be integrated on different time scales, improving stability and efficiency on stiff or anisotropic targets.

## What is here
- JAX 50D benchmark for SVGD variants vs SGLD/SGHMC, with gradient and kernel eval accounting.
- JAX 2D benchmark suite (banana, ring, squiggly, two_moons) with comparable metrics plus grid-based L1.
- Early stopping based on KSD to prevent late-run degradation (configurable in each benchmark).
- PyTorch legacy experiments and exploratory scripts (see `mri_samplers.py`, `experiments.py`, `misc/`).
- Diagnostics and design notes in `docs/ideas.md`.

## Key JAX files
- `jax/samplers.py`: SVGD variants (vanilla, Strang, fixed multirate, adaptive error-controlled multirate) plus SGLD/SGHMC.
- `jax/target_50d.py`: 50D Gaussian target with whitening matrix.
- `jax/targets_2d.py`: 2D targets and cached reference mean/cov via grid integration.
- `jax/metrics_50d.py`: mu error, cov error, ESS, KSD, mean log-prob (50D).
- `jax/metrics_2d.py`: cov error, ESS, KSD, mean log-prob (2D).

## 50D workflow
- Run: `python jax/benchmark_gauss50.py`
- Outputs: `metrics_50d/metrics_gauss50.csv`
- Plot: `python jax/plot_gauss50.py`
- Figures: `figures_50d/`

Notes:
- Dual-axis plots show grad evals (left) and kernel evals (right).
- ESS is shown as bars only.
- Toggle `USE_WHITENING` in `jax/benchmark_gauss50.py` to enable whitening for SVGD-family methods.
- Early stopping uses KSD with a tolerance/patience guard (`EARLY_STOP*` settings).

## 2D workflow
- Run: `python jax/benchmark_2d.py`
- Outputs: `metrics_2d/<target>.csv`
- Plot: `python jax/plot_2d.py`
- Figures: `figures_2d/<target>/`
- Animate: `python jax/animate_2d.py --target banana --sampler multirate_svgd --out animations_2d/banana_multirate.gif`

Notes:
- `plot_2d.py` auto-discovers all CSVs in `metrics_2d/` and writes per-target folders.
- `animate_2d.py` works for both particle methods and single-chain methods (one moving point).
- Early stopping uses the same KSD logic as 50D (`EARLY_STOP*` settings).

## How to extend or revise
- Add a new 2D target:
  1) Implement `logp` in `jax/targets_2d.py` and set bounds.
  2) Add the target name to `RUN_TARGETS` in `jax/benchmark_2d.py`.
  3) Re-run the benchmark and plot scripts.
- Add a new sampler:
  1) Implement in `jax/samplers.py`.
  2) Return `grad_evals` and `kernel_evals` in the `info` dict for fair comparisons.
  3) Register in `jax/benchmark_gauss50.py` and `jax/benchmark_2d.py`.
- Add or edit metrics in `jax/metrics_50d.py` or `jax/metrics_2d.py` and wire them into the benchmarks.

## Supporting material
- `misc/`: prototypes, old scripts, and one-off experiments.
- `Notebooks/`: exploratory notebooks.
- `figures_2d/`, `figures_50d/`, `animations_2d/`: generated outputs.
- `metrics_2d/`, `metrics_50d/`: benchmark CSVs (tracked with DVC in this repo).

## Current state
The repository is research-oriented and experimental. The multirate SVGD designs are under active exploration, with multiple implementations and diagnostics to compare stability, ESS, and error against baselines.
