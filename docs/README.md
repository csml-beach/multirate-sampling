# Benchmark Runbook

This document contains detailed commands and output locations for reproducing experiments and figures.

## Environment Setup

### Option A: GitHub Codespaces (recommended)

This repository includes a ready-to-use dev container:
- `.devcontainer/devcontainer.json`
- `.devcontainer/postCreate.sh`

Steps:
1. Open the repository in Codespaces.
2. Wait for post-create setup to complete.
3. Verify imports and JAX device detection:

```bash
python -c "import jax, optax; print(jax.devices())"
```

Recommended machine sizing:
- Small/medium benchmarks: default machine is usually sufficient.
- Full-scale HLR experiments: use a larger machine (for example 8 cores / 16 GB RAM or higher).

### Option B: Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- UCI datasets are downloaded automatically to `data/uci/` when first needed.
- Public reproduction only requires running the benchmark and plotting scripts in this repository.

## Methods

Method names used in code, plots, and manuscript tables:
- `SVGD`
- `Strang-SVGD`
- `MR-SVGD`
- `Adapt-MR-SVGD`
- `SGLD`
- `SGHMC`

## 50D Gaussian

- Run benchmark: `python jax/benchmarks/gauss50/benchmark_gauss50.py`
- Plot results: `python jax/benchmarks/gauss50/plot_gauss50.py`
- Metrics output: `metrics/50d/metrics_gauss50.csv`
- Figures output: `figures/50d/`

Notes:
- Dual-axis plots show gradient evaluations (left) and kernel evaluations (right).
- Early stopping uses KSD-based patience/tolerance settings in the benchmark script.
- Whitening can be toggled in `jax/benchmarks/gauss50/benchmark_gauss50.py`.

## 2D Targets

- Run benchmark: `python jax/benchmarks/2d/benchmark_2d.py`
- Plot results: `python jax/benchmarks/2d/plot_2d.py`
- Metrics output: `metrics/2d/<target>.csv`
- Figures output: `figures/2d/<target>/`

Animation example:

```bash
python jax/benchmarks/2d/animate_2d.py \
  --target banana \
  --sampler multirate_svgd \
  --out animations/2d/banana_multirate.gif
```

Notes:
- `plot_2d.py` auto-discovers CSV files under `metrics/2d/`.
- Both particle methods and chain-based methods are supported in animations.

## UCI Logistic Regression

- Run all datasets: `python jax/benchmarks/uci/benchmark_logreg.py`
- Run subset: `python jax/benchmarks/uci/benchmark_logreg.py --datasets a5a`
- Plot results: `python jax/benchmarks/uci/plot_uci.py`
- Metrics output: `metrics/uci/<dataset>.csv`
- Figures output: `figures/uci/<dataset>/`

Notes:
- Datasets: `breast_cancer`, `ionosphere`, `spambase`, `a5a`.
- NLL-based early stopping is used by default.

## UCI Bayesian Neural Network

- Run all datasets: `python jax/benchmarks/bnn/benchmark_bnn.py`
- Run subset: `python jax/benchmarks/bnn/benchmark_bnn.py --datasets a5a`
- Plot results: `python jax/benchmarks/bnn/plot_bnn.py`
- Metrics output: `metrics/bnn/<dataset>.csv`
- Figures output: `figures/bnn/<dataset>/`

Notes:
- Uses a one-hidden-layer Bayesian neural network.
- Default datasets: `breast_cancer`, `ionosphere`, `a5a`.
- KSD-based early stopping is enabled by default.

## 2D Mixture (mix8)

- Run benchmark: `python jax/benchmarks/mixture2d/benchmark_mixture2d.py`
- Plot results: `python jax/benchmarks/mixture2d/plot_mixture2d.py`
- Metrics output: `metrics/mixture2d/mix8.csv`
- Figures output: `figures/mixture2d/mix8/`

Animate all methods:

```bash
python jax/benchmarks/mixture2d/animate_mixture2d.py --target mix8 --all-methods
```

Notes:
- Metrics include mode coverage, mode entropy, min mass per mode, KSD, and grid L1.
- Benchmark uses KSD-based early stopping with best-checkpoint restoration.

## HLR (Hierarchical Logistic Regression)

- Run longtail (default): `python jax/benchmarks/hlr/benchmark_hlr.py --group-mode longtail`
- Run uniform: `python jax/benchmarks/hlr/benchmark_hlr.py --group-mode uniform`
- Run both: `python jax/benchmarks/hlr/benchmark_hlr.py --group-mode both`
- Plot results: `python jax/benchmarks/hlr/plot_hlr.py`
- Metrics output: `metrics/hlr/longtail.csv`, `metrics/hlr/uniform.csv`
- Figures output: `figures/hlr/longtail/`, `figures/hlr/uniform/`

Quick low-compute example:

```bash
python jax/benchmarks/hlr/benchmark_hlr.py \
  --group-mode longtail \
  --n-samples 4000 \
  --n-features 24 \
  --n-groups 300 \
  --iters 40 \
  --save-every 20 \
  --particles 12 \
  --seeds 0
```

Full-scale example:

```bash
python jax/benchmarks/hlr/benchmark_hlr.py \
  --group-mode longtail \
  --n-samples 1000000 \
  --n-features 300 \
  --n-groups 50000 \
  --iters 1000 \
  --save-every 20 \
  --particles 32 \
  --seeds 0,1,2,3,4
```

Notes:
- Early stopping is NLL-only with patience plus non-finite fail-fast checks.
- ECE is reported but not used as an early-stop trigger.

## Output Locations

- `metrics/`: benchmark CSV results.
- `figures/`: static figures.
- `animations/`: GIF animations.

## CI Smoke Test

- Workflow: `.github/workflows/smoke-test.yml`
- Trigger: `workflow_dispatch` (manual run from GitHub Actions tab)
- Scope: runs a short synthetic HLR benchmark and verifies expected methods are present in output CSV.
