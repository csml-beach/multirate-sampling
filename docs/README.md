# Multirate Sampling (SVGD-focused)

## Purpose
This repository explores multirate variants of particle-based sampling, with an initial focus on SVGD-style dynamics. The core idea is to split or reweight the repulsive (kernel) and attractive (log-density gradient) components of SVGD so they can be integrated on different time scales, with the goal of better stability and efficiency on stiff or anisotropic targets.

In practice, the codebase contains:
- JAX experiments that benchmark SVGD variants (vanilla, Strang-split, and a multirate-ratio variant) against SGLD/SGHMC on a 50D Gaussian with a wide eigenvalue spectrum.
- PyTorch experiments on 2D toy targets (banana, ring, squiggly, spiral, mixtures) with SVGD and multirate-inspired updates, including adaptive step control and flop-count tracking.
- Prototype and exploratory scripts for MIS/MRI ideas, classic MCMC baselines (MALA/HMC/Gibbs), and visualization/animation assets.

## Key JAX path (higher-dimensional benchmark)
- `jax/samplers.py`: SVGD variants including Strang splitting and a multirate ratio boost (with optional whitening).
- `jax/target.py`: constructs a random 50D Gaussian target and provides log-density and whitening matrix.
- `jax/benchmark_gauss50.py`: runs benchmark loops and records mean/covariance error and ESS to `metrics_gauss50.csv`.
- `jax/plot_gauss50.py`: generates plots in `figures/` from the benchmark CSV.

## Key PyTorch path (2D toy targets)
- `mri_samplers.py`: target families, SVGD/MRI-style samplers, flop counter, and metric tracking.
- `experiments.py`: runs and visualizes SVGD vs multirate-style samplers on selected 2D targets.
- `bayes-lin-reg-sampler.py`: standalone SVGD demo on a Bayesian nonlinear regression toy problem.

## Supporting and exploratory material
- `misc/`: prototype scripts for MIS/MRI ideas, SVGD variants, and MALA/HMC/Gibbs baselines.
- `Notebooks/`: exploratory notebooks for MIS/SVGD variations and error-control experiments.
- `animations/` and `figures/`: saved visualizations and animations from experiments.

## Current state
The repository is experimental and research-oriented. The multirate SVGD ideas are under active exploration, with multiple implementations and diagnostics to compare stability, ESS, and error against standard baselines.
