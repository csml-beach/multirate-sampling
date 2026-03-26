#!/usr/bin/env bash
set -euo pipefail

if command -v pyenv >/dev/null 2>&1; then
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  if [[ -d "$PYENV_ROOT" ]]; then
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    if pyenv commands | grep -q '^activate$'; then
      pyenv activate jax
    else
      pyenv shell jax
    fi
  fi
fi

python jax/benchmarks/mixture2d/plot_mixture2d.py
python jax/benchmarks/2d/plot_2d.py
python jax/benchmarks/uci/plot_uci.py
python jax/benchmarks/gauss50/plot_gauss50.py
python jax/benchmarks/bnn/plot_bnn.py
