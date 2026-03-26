#!/usr/bin/env bash
set -euo pipefail

if command -v pyenv >/dev/null 2>&1; then
  export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
  if [[ -d "$PYENV_ROOT" ]]; then
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
  fi
  pyenv shell jax
fi

dvc add figures animations metrics
dvc push

git add -u
git commit -m "snapshot at $(date '+%Y-%m-%d %H:%M:%S')"
git push
