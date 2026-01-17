# Multirate Sampling (SVGD-focused)

## Purpose
This repository explores multirate variants of particle-based sampling, with an initial focus on SVGD-style dynamics. The core idea is to split or reweight the repulsive (kernel) and attractive (log-density gradient) components of SVGD so they can be integrated on different time scales, improving stability and efficiency on stiff or anisotropic targets.

## What is here
- JAX 50D benchmark for SVGD variants vs SGLD/SGHMC, with gradient and kernel eval accounting.
- JAX 2D benchmark suite (banana, ring, squiggly, two_moons, funnel) with comparable metrics plus grid-based L1.
- JAX UCI logistic regression benchmark (breast_cancer, ionosphere, spambase, a5a) with accuracy/NLL/ECE/ESS/KSD metrics.
- JAX 2D mixture benchmark (mix8) with mode coverage/entropy + KSD and grid L1.
- Early stopping based on KSD to prevent late-run degradation (configurable in each benchmark).
- PyTorch legacy experiments and exploratory scripts (see `mri_samplers.py`, `experiments.py`, `misc/`).
- Diagnostics and design notes in `docs/ideas.md`.

## Key JAX files
- `jax/samplers.py`: SVGD variants (vanilla, Strang, fixed multirate, adaptive error-controlled multirate) plus SGLD/SGHMC.
- `jax/benchmarks/gauss50/target_50d.py`: 50D Gaussian target with whitening matrix.
- `jax/benchmarks/2d/targets_2d.py`: 2D targets and cached reference mean/cov via grid integration.
- `jax/benchmarks/gauss50/metrics_50d.py`: mu error, cov error, ESS, KSD, mean log-prob (50D).
- `jax/benchmarks/2d/metrics_2d.py`: cov error, ESS, KSD, mean log-prob (2D).
- `jax/benchmarks/uci/benchmark_logreg.py`: UCI logistic regression benchmark (multi-dataset, early stop, seed ablation).
- `jax/benchmarks/uci/datasets.py`: dataset loaders (WDBC, ionosphere, spambase, a5a).
- `jax/benchmarks/uci/metrics_uci.py`: accuracy, NLL, ECE, ESS, KSD, mean log-prob (UCI).
- `jax/benchmarks/mixture2d/`: mixture benchmark (targets, metrics, plots, animations).

## 50D workflow
- Run: `python jax/benchmarks/gauss50/benchmark_gauss50.py`
- Outputs: `metrics/50d/metrics_gauss50.csv`
- Plot: `python jax/benchmarks/gauss50/plot_gauss50.py`
- Figures: `figures/50d/`

Notes:
- Dual-axis plots show grad evals (left) and kernel evals (right).
- ESS is shown as bars only.
- Toggle `USE_WHITENING` in `jax/benchmarks/gauss50/benchmark_gauss50.py` to enable whitening for SVGD-family methods.
- Early stopping uses KSD with a tolerance/patience guard (`EARLY_STOP*` settings).
- Seed ablation is supported via `SEEDS`, and best checkpoints are appended (`is_best=1`) for summary plots.

## 2D workflow
- Run: `python jax/benchmarks/2d/benchmark_2d.py`
- Outputs: `metrics/2d/<target>.csv`
- Plot: `python jax/benchmarks/2d/plot_2d.py`
- Figures: `figures/2d/<target>/`
- Animate: `python jax/benchmarks/2d/animate_2d.py --target banana --sampler multirate_svgd --out animations/2d/banana_multirate.gif`

Notes:
- `plot_2d.py` auto-discovers all CSVs in `metrics/2d/` and writes per-target folders.
- `animate_2d.py` works for both particle methods and single-chain methods (one moving point).
- Early stopping uses the same KSD logic as 50D (`EARLY_STOP*` settings).

## UCI workflow (logistic regression)
- Run all datasets: `python jax/benchmarks/uci/benchmark_logreg.py`
- Run a subset: `python jax/benchmarks/uci/benchmark_logreg.py --datasets a5a`
- Outputs: `metrics/uci/<dataset>.csv`
- Plot: `python jax/benchmarks/uci/plot_uci.py`
- Figures: `figures/uci/<dataset>/`

Notes:
- Metrics are computed on the test split; the dataset seed controls the train/test split.
- Summary plots use mean ± std across seeds and include a 2x2 panel (NLL, ECE, Accuracy, ESS).
- A speed-vs-quality scatter shows final NLL/ECE vs wall time with error bars.

## Mixture2D workflow
- Run: `python jax/benchmarks/mixture2d/benchmark_mixture2d.py`
- Outputs: `metrics/mixture2d/mix8.csv`
- Plot: `python jax/benchmarks/mixture2d/plot_mixture2d.py`
- Figures: `figures/mixture2d/mix8/`
- Animate all methods: `python jax/benchmarks/mixture2d/animate_mixture2d.py --target mix8 --all-methods`

Notes:
- Metrics include mode coverage, mode entropy, min mass per mode, KSD, and grid L1.
- Benchmark uses KSD-based early stopping with a restore-best row (`is_best=1`).

## How to extend or revise
- Add a new 2D target:
  1) Implement `logp` in `jax/benchmarks/2d/targets_2d.py` and set bounds.
  2) Add the target name to `RUN_TARGETS` in `jax/benchmarks/2d/benchmark_2d.py`.
  3) Re-run the benchmark and plot scripts.
- Add a new sampler:
  1) Implement in `jax/samplers.py`.
  2) Return `grad_evals` and `kernel_evals` in the `info` dict for fair comparisons.
  3) Register in `jax/benchmarks/gauss50/benchmark_gauss50.py` and `jax/benchmarks/2d/benchmark_2d.py`.
- Add or edit metrics in `jax/benchmarks/gauss50/metrics_50d.py` or `jax/benchmarks/2d/metrics_2d.py` and wire them into the benchmarks.

## Supporting material
- `misc/`: prototypes, old scripts, and one-off experiments.
- `Notebooks/`: exploratory notebooks.
- `figures/2d/`, `figures/50d/`, `animations/2d/`: generated outputs.
- `metrics/2d/`, `metrics/50d/`: benchmark CSVs (tracked with DVC in this repo).
- `metrics/uci/`, `figures/uci/`: UCI benchmark outputs (tracked with DVC).

## Current state
The repository is research-oriented and experimental. The multirate SVGD designs are under active exploration, with multiple implementations and diagnostics to compare stability, ESS, and error against baselines.
